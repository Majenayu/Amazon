#!/usr/bin/env python3
"""Measure blocking recall against ground truth: what fraction of true links did
Stage 1 actually retrieve? This is the ceiling on the final score.

Reports both micro link recall and the per-entity ALL-retained rate (the number that
actually bounds macro F_0.5, per red-team D7).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="artifacts/sample_candidates")
    ap.add_argument("--ground-truth", default="data/sample/train/train_ground_truth.tsv")
    args = ap.parse_args()

    # Load candidates: {s1_id: set(tgt_id)}
    cand = {}
    for f in sorted(glob.glob(os.path.join(args.candidates, "*.parquet"))):
        df = pd.read_parquet(f, columns=["s1_id", "tgt_id"])
        for s1, tgt in zip(df["s1_id"], df["tgt_id"]):
            cand.setdefault(s1, set()).add(tgt)

    gt = pd.read_csv(args.ground_truth, sep="\t", dtype=str,
                     keep_default_na=False, na_filter=False)

    total_links = retrieved_links = 0
    n_matched_entities = all_retained = 0
    n_singletons = 0
    for s1, mids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        true_set = {x for x in mids.split(",") if x} if mids.strip() else set()
        if not true_set:
            n_singletons += 1
            continue
        n_matched_entities += 1
        got = cand.get(s1, set())
        hit = len(true_set & got)
        total_links += len(true_set)
        retrieved_links += hit
        if hit == len(true_set):
            all_retained += 1

    print("Blocking recall report")
    print("  matched entities      : %d" % n_matched_entities)
    print("  singletons            : %d" % n_singletons)
    print("  micro link recall     : %.2f%% (%d / %d)"
          % (100.0 * retrieved_links / max(1, total_links), retrieved_links, total_links))
    print("  per-entity ALL-retained: %.2f%% (%d / %d)  <- caps macro F_0.5"
          % (100.0 * all_retained / max(1, n_matched_entities),
             all_retained, n_matched_entities))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
