#!/usr/bin/env python3
"""Apply the adaptive per-entity decision rule instead of a global threshold.

Writes exactly the same two submission files as predict.py, so it is a drop-in
replacement for stage 4:

    python src/baseline/decide.py --pairs data/cache/v3_test.tsv \
        --model models/v3 --s1 data/test/test_source1.tsv \
        --matching-out output/matching_results.tsv \
        --candidate-out output/candidate_pairs.tsv

Memory design matches predict.py: pairs are streamed, predictions are bucketed
to disk by hash(source1_id), and each bucket is decided in turn, so peak memory
does not grow with the size of the pairs file.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from collections import defaultdict

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))

from baseline.decision import expected_score  # noqa: E402
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


def main() -> int:
    ap = argparse.ArgumentParser(description="adaptive per-entity decision")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model", required=True, help="model prefix (no extension)")
    ap.add_argument("--s1", required=True)
    ap.add_argument("--matching-out", default="output/matching_results.tsv")
    ap.add_argument("--candidate-out", default="output/candidate_pairs.tsv")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--m-prior", type=float, default=1.0,
                    help=">1 makes the rule more willing to predict; tuned on train")
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
    thr = float(bundle.get("threshold", 0.5))
    nb = args.buckets
    log("model loaded | features=%d | global threshold=%.4f | m_prior=%.3f"
        % (len(feats), thr, args.m_prior))

    s1_ids = read_s1_ids(args.s1)
    log("Source-1 rows to emit: {:,}".format(len(s1_ids)))

    if args.out_dir:
        args.matching_out = os.path.join(args.out_dir, "matching_results.tsv")
        args.candidate_out = os.path.join(args.out_dir, "candidate_pairs.tsv")
    out_dir = os.path.dirname(os.path.abspath(args.matching_out))
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = os.path.join(out_dir, "_decide_buckets")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir)
    handles = [open(os.path.join(tmp_dir, "b%02d.tsv" % i), "w", encoding="utf-8",
                     newline="\n") for i in range(nb)]

    # ---- phase A: score every pair once, spill (s1_id, other_id, prob) -------
    n_rows = 0
    t0 = time.time()
    reader = pd.read_csv(args.pairs, sep="\t", chunksize=args.chunksize,
                         dtype={"s1_id": str, "other_id": str})
    for chunk in reader:
        probs = clf.predict_proba(chunk[feats].to_numpy(dtype=np.float32))[:, 1]
        s1c = chunk["s1_id"].to_numpy()
        oc = chunk["other_id"].to_numpy()
        for s1, o, p in zip(s1c, oc, probs):
            handles[hash(s1) & (nb - 1)].write("%s\t%s\t%.6f\n" % (s1, o, p))
        n_rows += len(chunk)
        log("  scored {:,} pairs in {:.0f}s".format(n_rows, time.time() - t0))
    for h in handles:
        h.close()
    if n_rows == 0:
        log("no candidate pairs found - nothing to predict")
        return 1

    # ---- phase B: decide bucket by bucket ----------------------------------
    pred_map = {}
    cand_of = {}
    stats = defaultdict(int)
    t1 = time.time()
    for b in range(nb):
        by_entity = defaultdict(list)
        with open(os.path.join(tmp_dir, "b%02d.tsv" % b), encoding="utf-8") as fh:
            for line in fh:
                s1, o, p = line.rstrip("\n").split("\t")
                by_entity[s1].append((float(p), o))
        for s1, lst in by_entity.items():
            lst.sort(key=lambda t: -t[0])
            cand_of[s1] = [o for _, o in lst]
            k, ev, base = expected_score([p for p, _ in lst], args.m_prior)
            if k <= 0:
                stats["empty"] += 1
            else:
                stats["nonempty"] += 1
                stats["pairs"] += k
                if ev > base + 1e-9:
                    stats["helped"] += 1
                else:
                    stats["hurt"] += 1
            pred_map[s1] = [o for _, o in lst[:k]]
        by_entity.clear()
        if (b + 1) % 8 == 0:
            log("  decided bucket %d/%d in %.0f" % (b + 1, nb, time.time() - t1))
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ---- phase C: write the two submission files ---------------------------
    emitted = set()
    n_nonempty = total_pred = appended = 0
    with open(args.matching_out, "w", encoding="utf-8", newline="\n") as mout, \
            open(args.candidate_out, "w", encoding="utf-8", newline="\n") as cout:
        mout.write("source1_entity_id\tmatched_entity_ids\n")
        cout.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in s1_ids:
            lst = pred_map.get(sid)
            if lst:
                mout.write(sid + "\t" + ",".join(lst) + "\n")
                n_nonempty += 1
                total_pred += len(lst)
            else:
                mout.write(sid + "\t\n")
            cl = cand_of.get(sid)
            if cl:
                cout.write(sid + "\t" + ",".join(cl) + "\n")
                emitted.add(sid)
            else:
                cout.write(sid + "\t\n")
                appended += 1

    log("")
    log("=== output ===")
    log("  matching_results.tsv : {:,} entities | {:,} non-empty | {:,} matches total"
        .format(len(s1_ids), n_nonempty, total_pred))
    log("  candidate_pairs.tsv  : {:,} entities with candidates (+ {:,} empty)"
        .format(len(emitted), appended))
    log("  decision split       : {:,} predicted EMPTY (singleton), {:,} non-empty"
        .format(stats["empty"], stats["nonempty"]))
    log("  -> %s" % args.matching_out)
    log("  -> %s" % args.candidate_out)
    log("  done in %.0f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())