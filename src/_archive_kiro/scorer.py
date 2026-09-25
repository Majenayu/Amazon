#!/usr/bin/env python3
"""Exact macro F_0.5 scorer, including singletons (red-team S0.1).

F_0.5 per S1 entity: 1.25*TP / (1.25*TP + FP + 0.25*FN).
  m=0, k=0 -> 1.0 ;  m=0, k>0 -> 0.0.
Averaged over ALL S1 entities in the evaluation set.

Used only for local validation against known labels; never touches the test set.
"""

from __future__ import annotations

import argparse

import pandas as pd


def f_half(true_set: set, pred_set: set) -> float:
    m, k = len(true_set), len(pred_set)
    if m == 0:
        return 1.0 if k == 0 else 0.0
    tp = len(true_set & pred_set)
    fp = k - tp
    fn = m - tp
    denom = 1.25 * tp + fp + 0.25 * fn
    return (1.25 * tp) / denom if denom else 0.0


def macro_f_half(pred_map: dict, gt_map: dict) -> float:
    """pred_map/gt_map: {s1_id: set(tgt_id)}. Averaged over all gt entities."""
    if not gt_map:
        return 0.0
    total = 0.0
    for s1, true_set in gt_map.items():
        total += f_half(true_set, pred_map.get(s1, set()))
    return total / len(gt_map)


def _load_results(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False)
    col = df.columns[1]
    out = {}
    for s1, ids in zip(df[df.columns[0]], df[col]):
        out[s1] = {x for x in ids.split(",") if x} if ids.strip() else set()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--ground-truth", required=True)
    args = ap.parse_args()
    pred = _load_results(args.pred)
    gt = _load_results(args.ground_truth)
    score = macro_f_half(pred, gt)
    print("macro F_0.5 (incl. singletons): %.4f over %d entities" % (score, len(gt)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
