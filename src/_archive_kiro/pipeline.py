#!/usr/bin/env python3
"""End-to-end entity-resolution pipeline for the Amazon ML Challenge 2026.

Stages
------
  0. build_maps  - derive label-free state canonicalization maps from the files.
  1. blocking    - forward-primary IDF retrieval -> candidate_pairs (per-S1 floor).
  2a. features   - turn candidate pairs into numeric feature shards.
  2b. matcher    - train LightGBM on TRAIN features; score candidate pairs.
  3. decide      - accept per entity + one-owner resolution -> matching_results.tsv.

Two modes
---------
  train : run stages 0,1,2a on the TRAIN split, then train the matcher. Also scores
          the train candidates and reports macro F_0.5 so we know where we stand.
  test  : run stages 0,1,2a on the TEST split, score with the trained model, decide,
          and write output/matching_results.tsv + output/candidate_pairs.tsv for EVERY
          test S1 entity (US, India, France) - none dropped.

Memory-safe throughout: sources streamed in chunks; candidate/feature/score data
written as parquet shards; nothing holds the full pair matrix in RAM at once. Targets
Camber SMALL (64 GB, 16 cores) but runs on smaller nodes with smaller --top-k.

Usage
-----
  python pipeline.py train --data-dir data
  python pipeline.py test  --data-dir data
  # quick local check on the bundled sample:
  python pipeline.py train --data-dir data/sample
  python pipeline.py test  --data-dir data/sample
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import build_maps          # noqa: E402
import blocking            # noqa: E402
import features            # noqa: E402
import matcher             # noqa: E402
import decide              # noqa: E402
import scorer              # noqa: E402


def _paths(data_dir, split):
    base = os.path.join(data_dir, split)
    return {
        "s1": os.path.join(base, "%s_source1.tsv" % split),
        "s2": os.path.join(base, "%s_source2.tsv" % split),
        "s3": os.path.join(base, "%s_source3.tsv" % split),
        "gt": os.path.join(base, "train_ground_truth.tsv"),
    }


def run_common(data_dir, split, art, top_k, df_cap, state_map):
    """Stages 1 + 2a for a split. Returns (cand_dir, feat_dir, token_to_id, idf)."""
    p = _paths(data_dir, split)
    targets = [p["s2"], p["s3"]]

    cand_dir = os.path.join(art, "%s_candidates" % split)
    _, idx, token_to_id, idf = blocking.generate_candidates(
        p["s1"], targets, cand_dir, top_k=top_k, df_cap=df_cap)

    print("loading records for features ...", file=sys.stderr)
    s1_records = features.load_records(p["s1"], state_map)
    tgt_records = features.load_records(p["s2"], state_map)
    tgt_records.update(features.load_records(p["s3"], state_map))

    gt_map = None
    if split == "train" and os.path.isfile(p["gt"]):
        gt_map = features.load_gt_map(p["gt"])

    feat_dir = os.path.join(art, "%s_features" % split)
    features.build_features(cand_dir, s1_records, tgt_records, token_to_id, idf,
                            feat_dir, gt_map=gt_map)
    return cand_dir, feat_dir, token_to_id, idf


def cmd_train(args):
    art = args.artifacts
    os.makedirs(art, exist_ok=True)

    # Stage 0: label-free maps from all files.
    state_map_path = os.path.join(art, "state_maps.json")
    build_maps.build([os.path.join(args.data_dir, "train"),
                      os.path.join(args.data_dir, "test")],
                     state_map_path, min_shared=2)
    state_map = build_maps.load_state_maps(state_map_path)

    _, feat_dir, _, _ = run_common(args.data_dir, "train", art,
                                   args.top_k, args.df_cap, state_map)

    model_path = os.path.join(art, "model.txt")
    matcher.train(feat_dir, model_path, num_round=args.num_round)

    # Self-check: score train candidates and report macro F_0.5.
    score_dir = os.path.join(art, "train_scores")
    matcher.score(feat_dir, model_path, score_dir)
    by_s1 = decide.load_scores(score_dir)
    required = decide.read_required_s1(_paths(args.data_dir, "train")["s1"])
    final = decide.decide(by_s1, required, threshold=args.threshold,
                          best_floor=args.best_floor)
    pred_map = {s1: set(v) for s1, v in final.items()}
    gt_map = features.load_gt_map(_paths(args.data_dir, "train")["gt"])
    macro = scorer.macro_f_half(pred_map, gt_map)
    print("\n=== TRAIN macro F_0.5 (in-sample, optimistic): %.4f ===" % macro)


def cmd_test(args):
    art = args.artifacts
    state_map_path = os.path.join(art, "state_maps.json")
    if not os.path.isfile(state_map_path):
        build_maps.build([os.path.join(args.data_dir, "train"),
                          os.path.join(args.data_dir, "test")],
                         state_map_path, min_shared=2)
    state_map = build_maps.load_state_maps(state_map_path)

    cand_dir, feat_dir, _, _ = run_common(args.data_dir, "test", art,
                                          args.top_k, args.df_cap, state_map)

    model_path = os.path.join(art, "model.txt")
    if not os.path.isfile(model_path):
        sys.exit("No trained model at %s - run 'train' first." % model_path)
    score_dir = os.path.join(art, "test_scores")
    matcher.score(feat_dir, model_path, score_dir)

    by_s1 = decide.load_scores(score_dir)
    required = decide.read_required_s1(_paths(args.data_dir, "test")["s1"])
    final = decide.decide(by_s1, required, threshold=args.threshold,
                          best_floor=args.best_floor)

    os.makedirs(args.output, exist_ok=True)
    decide.write_results(final, required,
                         os.path.join(args.output, "matching_results.tsv"))
    decide.write_candidates(by_s1, required,
                            os.path.join(args.output, "candidate_pairs.tsv"))
    print("\nDone. Every one of %d test S1 entities has a row (France included)."
          % len(required))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["train", "test"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--output", default="output")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--df-cap", type=int, default=40_000)
    ap.add_argument("--num-round", type=int, default=2000)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--best-floor", type=float, default=0.30)
    args = ap.parse_args()

    if args.mode == "train":
        cmd_train(args)
    else:
        cmd_test(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
