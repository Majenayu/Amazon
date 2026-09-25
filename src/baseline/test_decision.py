#!/usr/bin/env python3
"""Does the adaptive decision rule actually beat a global threshold?

Run on the real labelled training pairs:

    python src/baseline/test_decision.py --pairs data/cache/v3_train.tsv \
        --model models/v3 --gt data/train/train_ground_truth.tsv

Compares, on the same held-out entities at the same candidate density:
  * the predict-all-empty floor,
  * the best single global threshold (what stage 4 does today),
  * the adaptive per-entity rule, swept over --priors.

Exit code 0 if the adaptive rule wins, 1 if it does not.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zlib
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))

from baseline.decision import expected_score  # noqa: E402
from baseline.textnorm import FEATURE_NAMES  # noqa: E402


def log(m):
    print(m, file=sys.stderr, flush=True)


def split_of(s1_id: str) -> str:
    return "val" if (zlib.crc32(s1_id.encode("utf-8")) % 5 == 0) else "trn"


def macro_score(pred: dict, truth: dict) -> float:
    """The exact competition metric, averaged over every validation entity."""
    tot = 0.0
    for sid, mset in truth.items():
        got = pred.get(sid, ())
        m, k = len(mset), len(got)
        if m == 0 and k == 0:
            s = 1.0
        elif m == 0:
            s = 0.0
        else:
            s = 1.25 * len(set(got) & mset) / (0.25 * m + k)
        tot += s
    return tot / max(len(truth), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--priors", default="0.85,1.0,1.15,1.3,1.6")
    ap.add_argument("--max-val-entities", type=int, default=0,
                    help="cap validation entities for a fast check (0 = all)")
    args = ap.parse_args()

    import joblib
    bundle = joblib.load(args.model + ".joblib")
    clf = bundle["model"]
    feats = bundle.get("features", FEATURE_NAMES)

    # ---- ground truth of the validation entities --------------------------
    truth = {}
    with open(args.gt, encoding="utf-8", errors="replace") as fh:
        next(fh, None)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) > 1 and split_of(p[0]) == "val":
                truth[p[0]] = set(x for x in p[1].split(",") if x)
    if args.max_val_entities:
        truth = dict(list(truth.items())[:args.max_val_entities])
    log("validation entities: {:,}".format(len(truth)))
    floor = sum(1.0 if not m else 0.0 for m in truth.values()) / max(len(truth), 1)
    log("predict-all-empty floor : {:.4f}".format(floor))

    # ---- one streaming pass: candidates + probabilities, kept aligned ------
    # ent[s1] = list of (other_id, prob)
    ent = defaultdict(list)
    t0 = time.time()
    n = 0
    reader = pd.read_csv(args.pairs, sep="\t", chunksize=args.chunksize,
                         dtype={"s1_id": str, "other_id": str})
    for chunk in reader:
        s1col = chunk["s1_id"]
        mask = np.fromiter((split_of(x) == "val" for x in s1col), bool, len(chunk))
        if args.max_val_entities:
            mask &= s1col.isin(truth).to_numpy()
        sub = chunk[mask]
        if sub.empty:
            continue
        p = clf.predict_proba(sub[feats].to_numpy(dtype=np.float32))[:, 1]
        for s1, o, pp in zip(sub["s1_id"].to_numpy(), sub["other_id"].to_numpy(), p):
            ent[s1].append((o, float(pp)))
        n += len(sub)
    log("scored {:,} candidate pairs across {:,} entities in {:.0f}s"
        .format(n, len(ent), time.time() - t0))
    if not ent:
        log("no validation pairs - run the blocking stage first")
        return 1
    for s in ent:
        ent[s].sort(key=lambda t: -t[1])

    allp = np.array([p for lst in ent.values() for _, p in lst])

    # ---- baseline: best single global threshold ----------------------------
    best_t, best_s = 1.0, -1.0
    for t in np.unique(np.round(np.quantile(allp, np.linspace(0, 1, 121)), 4)):
        pred = {s: [o for o, p in lst if p >= t] for s, lst in ent.items()}
        sc = macro_score(pred, truth)
        if sc > best_s:
            best_s, best_t = sc, float(t)
    log("")
    log("=== global threshold (current stage-4 behaviour) ===")
    log("  best threshold   : {:.4f}".format(best_t))
    log("  best macro F0.5  : {:.4f}".format(best_s))

    # ---- adaptive per-entity rule ------------------------------------------
    log("")
    log("=== adaptive per-entity rule ===")
    best_p, best_ps = None, -1.0
    for prior in [float(x) for x in args.priors.split(",")]:
        pred = {}
        for s, lst in ent.items():
            k, _, _ = expected_score([p for _, p in lst], prior)
            pred[s] = [o for o, p in lst[:k]]
        sc = macro_score(pred, truth)
        log("  m_prior={:.2f}  macro F0.5 = {:.4f}".format(prior, sc))
        if sc > best_ps:
            best_ps, best_p = sc, prior

    log("")
    log("=== summary ===")
    log("  predict-all-empty  : {:.4f}".format(floor))
    log("  best global t      : {:.4f}  (t={:.4f})".format(best_s, best_t))
    log("  adaptive           : {:.4f}  (m_prior={})".format(best_ps, best_p))
    log("  gain from adaptive : {:+.4f}".format(best_ps - best_s))
    if best_ps - best_s > 1e-4:
        log("")
        log("VERDICT: the adaptive rule wins - use decide.py for stage 4.")
        return 0
    log("")
    log("VERDICT: no gain - keep the global threshold in predict.py.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())