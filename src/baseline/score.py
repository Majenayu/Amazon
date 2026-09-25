#!/usr/bin/env python3
"""Score a matching_results.tsv against ground truth with the official metric.

Use this on a *train* (or holdout) prediction to get the exact leaderboard number
before uploading anything. Entities present in the ground truth but missing from
the prediction file are counted as predicted-empty (1.0 if they are singletons,
0.0 otherwise) — the same rule the scorer uses.

Usage:
    python src/baseline/score.py --pred output/train_run/matching_results.tsv \\
        --gt data/sample/train/train_ground_truth.tsv
"""

from __future__ import annotations

import argparse
import sys

EMPTY = frozenset()


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def read_pred(path: str) -> dict:
    pred = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        next(fh, None)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            sid = p[0]
            if len(p) > 1 and p[1]:
                pred[sid] = frozenset(x for x in p[1].split(",") if x)
            else:
                pred[sid] = EMPTY
    return pred


def read_gt(path: str) -> dict:
    gt = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        next(fh, None)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) > 1 and p[1]:
                gt[p[0]] = frozenset(x for x in p[1].split(",") if x)
            else:
                gt[p[0]] = EMPTY
    return gt


def main() -> int:
    ap = argparse.ArgumentParser(description="Macro F_0.5 scorer.")
    ap.add_argument("--pred", required=True, help="matching_results.tsv")
    ap.add_argument("--gt", required=True, help="ground truth TSV")
    args = ap.parse_args()

    gt = read_gt(args.gt)
    pred = read_pred(args.pred)

    total = 0.0
    singletons = 0
    exact = 0
    wrong_empty = 0
    wrong_merge = 0
    for sid, true in gt.items():
        k_pred = pred.get(sid, EMPTY)
        m, k = len(true), len(k_pred)
        if m == 0:
            singletons += 1
            if k == 0:
                total += 1.0
                exact += 1
            else:
                wrong_merge += 1
        else:
            if k == 0:
                wrong_empty += 1
            else:
                t = len(k_pred & true)
                total += (1.25 * t) / (0.25 * m + k)
                if k_pred == true:
                    exact += 1

    n = len(gt)
    score = total / max(1, n)
    log("entities scored     : {:,}".format(n))
    log("macro F_0.5          : {:.4f}".format(score))
    log("perfect entities     : {:,} ({:.2f}%)".format(exact, 100.0 * exact / max(1, n)))
    log("singletons           : {:,} ({:.2f}%)".format(singletons, 100.0 * singletons / max(1, n)))
    log("singleton false merge: {:,}   <- each one scores 0.0".format(wrong_merge))
    log("missed entirely      : {:,}   <- predicted empty, had matches".format(wrong_empty))
    missing = sum(1 for sid in gt if sid not in pred)
    if missing:
        log("entities missing from prediction file: {:,}".format(missing))
    print("{:.4f}".format(score))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
