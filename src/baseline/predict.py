#!/usr/bin/env python3
"""Turn candidate pairs + a trained model into the two submission files.

    matching_results.tsv   source1_entity_id <TAB> matched_entity_ids (comma list)
    candidate_pairs.tsv    source1_entity_id <TAB> candidate_entity_ids (comma list)

EVERY Source-1 row appears in both files, even with an empty list - an entity
missing from matching_results.tsv is scored as zero, so we always emit all.

Memory design for the full test set (1.73M entities, ~35M candidate pairs):
we never hold all candidate lists in RAM. Candidates are bucketed to disk by
hash(source1 id), then each bucket is grouped and written in turn. Peak memory
stays around 1 GB no matter how big the pairs file is.

Usage:
    python src/baseline/predict.py --pairs data/cache/test_pairs.tsv \\
        --model models/full --s1 data/test/test_source1.tsv --out-dir output
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))

from baseline.textnorm import FEATURE_NAMES  # noqa: E402


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def read_s1_ids(path: str) -> list:
    ids = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        next(fh, None)
        for line in fh:
            ids.append(line.split("\t", 1)[0])
    return ids


def bucket_of(s1_id: str, nb: int) -> int:
    # builtin hash is fine: bucketing and reading happen inside one process
    return hash(s1_id) & (nb - 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Predict matches and write submission files.")
    ap.add_argument("--pairs", required=True, help="pairs TSV from block.py --split test")
    ap.add_argument("--model", required=True, help="model prefix from train.py (no extension)")
    ap.add_argument("--s1", required=True, help="test_source1.tsv")
    ap.add_argument("--out-dir", default="output",
                    help="folder for matching_results.tsv + candidate_pairs.tsv")
    ap.add_argument("--matching-out", default=None,
                    help="explicit path for matching_results.tsv (overrides --out-dir)")
    ap.add_argument("--candidate-out", default=None,
                    help="explicit path for candidate_pairs.tsv (overrides --out-dir)")
    ap.add_argument("--threshold", type=float, default=None, help="override tuned threshold")
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--buckets", type=int, default=64)
    args = ap.parse_args()

    import joblib
    for p in (args.pairs, args.s1, args.model + ".joblib"):
        if not os.path.exists(p):
            log("missing file: %s" % p)
            return 1

    bundle = joblib.load(args.model + ".joblib")
    clf = bundle["model"]
    feats = bundle.get("features", FEATURE_NAMES)
    thr = args.threshold if args.threshold is not None else float(bundle["threshold"])
    nb = args.buckets
    log("model loaded | threshold = %.4f" % thr)

    s1_ids = read_s1_ids(args.s1)
    log("Source-1 rows to emit: {:,}".format(len(s1_ids)))

    out_dir = os.path.dirname(os.path.abspath(args.matching_out)) if args.matching_out \
        else args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = os.path.join(out_dir, "_buckets")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir)
    handles = [open(os.path.join(tmp_dir, "b%02d.tsv" % i), "w", encoding="utf-8",
                    newline="\n") for i in range(nb)]

    # ---- phase A: score pairs, keep predictions, bucket every candidate -------
    pred_map = {}
    n_rows = n_keep = 0
    t0 = time.time()
    reader = pd.read_csv(args.pairs, sep="\t", chunksize=args.chunksize,
                         dtype={"s1_id": str, "other_id": str})
    for chunk in reader:
        probs = clf.predict_proba(chunk[feats].to_numpy(dtype=np.float32))[:, 1]
        keep = probs >= thr
        n_keep += int(keep.sum())
        n_rows += len(chunk)
        s1c = chunk["s1_id"].to_numpy()
        oc = chunk["other_id"].to_numpy()
        for s1, o in zip(s1c[keep], oc[keep]):
            lst = pred_map.get(s1)
            if lst is None:
                pred_map[s1] = [o]
            else:
                lst.append(o)
        for s1, o in zip(s1c, oc):
            handles[hash(s1) & (nb - 1)].write(s1 + "\t" + o + "\n")
        log("  scored {:,} pairs ({:,} above threshold) in {:.0f}s".format(
            n_rows, n_keep, time.time() - t0))
    for h in handles:
        h.close()
    if n_rows == 0:
        log("no candidate pairs found - nothing to predict")
        return 1

    # ---- phase B: group candidates bucket by bucket (bounded memory) ---------
    cand_path = args.candidate_out or os.path.join(out_dir, "candidate_pairs.tsv")
    emitted = set()
    t1 = time.time()
    with open(cand_path, "w", encoding="utf-8", newline="\n") as out:
        out.write("source1_entity_id\tcandidate_entity_ids\n")
        for b in range(nb):
            grouped = {}
            with open(os.path.join(tmp_dir, "b%02d.tsv" % b), encoding="utf-8") as fh:
                for line in fh:
                    s1, o = line.rstrip("\n").split("\t")
                    lst = grouped.get(s1)
                    if lst is None:
                        grouped[s1] = [o]
                    else:
                        lst.append(o)
            for s1, lst in grouped.items():
                out.write(s1 + "\t" + ",".join(lst) + "\n")
                emitted.add(s1)
            del grouped
    shutil.rmtree(tmp_dir, ignore_errors=True)
    log("candidate_pairs.tsv: {:,} entities with candidates in {:.0f}s".format(
        len(emitted), time.time() - t1))

    # ---- phase C: matching_results in Source-1 file order (+ empty rows) ------
    match_path = args.matching_out or os.path.join(out_dir, "matching_results.tsv")
    n_nonempty = total_pred = appended = 0
    with open(match_path, "w", encoding="utf-8", newline="\n") as mout, \
            open(cand_path, "a", encoding="utf-8", newline="\n") as cout:
        mout.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_ids:
            lst = pred_map.get(sid)
            if lst:
                mout.write(sid + "\t" + ",".join(lst) + "\n")
                n_nonempty += 1
                total_pred += len(lst)
            else:
                mout.write(sid + "\t\n")
            if sid not in emitted:
                cout.write(sid + "\t\n")
                appended += 1

    log("")
    log("=== output ===")
    log("  matching_results.tsv : {:,} entities | {:,} non-empty | {:,} matches total"
        .format(len(s1_ids), n_nonempty, total_pred))
    log("  candidate_pairs.tsv  : {:,} entities with candidates (+ {:,} empty)"
        .format(len(emitted), appended))
    log("  -> %s" % match_path)
    log("  -> %s" % cand_path)
    log("  done in %.0fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


