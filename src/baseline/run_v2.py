#!/usr/bin/env python3
"""Full v2 run: preparse once, block train, train model, block test, score."""
from __future__ import annotations
import argparse, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))
ROOT = os.path.dirname(os.path.dirname(HERE))
from baseline.hashjoin import read_gt
from baseline.v2block import run as v2run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=100_000)
    ap.add_argument("--tok-top", type=int, default=8)
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--resume-train", default="",
                    help="reuse existing hj pairs file for slices already done")
    a = ap.parse_args()
    data = os.path.join(ROOT, "data")
    pre = os.path.join(ROOT, "data", "cache", "pre")
    os.makedirs(pre, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
    t_all = time.time()
    train_pairs = os.path.join(ROOT, "data", "cache", "v2_train.tsv")
    print("[v2run] train blocking...", file=sys.stderr, flush=True)
    gt = read_gt(os.path.join(ROOT, "data", "train", "train_ground_truth.tsv"), None)
    v2run("train", data, pre, train_pairs, a.chunk, a.tok_top, gt=gt)
    rc = subprocess.call([sys.executable, os.path.join(ROOT, "src", "baseline", "train.py"),
                          "--pairs", train_pairs,
                          "--s1", os.path.join(ROOT, "data", "train", "train_source1.tsv"),
                          "--gt", os.path.join(ROOT, "data", "train", "train_ground_truth.tsv"),
                          "--model-out", os.path.join(ROOT, "models", "v2"),
                          "--max-iter", "100"], cwd=ROOT)
    if rc or a.train_only:
        return rc
    test_pairs = os.path.join(ROOT, "data", "cache", "v2_test.tsv")
    print("[v2run] test blocking...", file=sys.stderr, flush=True)
    v2run("test", data, pre, test_pairs, a.chunk, a.tok_top)
    rc = subprocess.call([sys.executable, os.path.join(ROOT, "src", "baseline", "predict.py"),
                          "--pairs", test_pairs,
                          "--model", os.path.join(ROOT, "models", "v2.joblib"),
                          "--test-dir", os.path.join(ROOT, "data", "test"),
                          "--matching-out", os.path.join(ROOT, "output", "matching_results.tsv"),
                          "--candidate-out", os.path.join(ROOT, "output", "candidate_pairs.tsv")],
                         cwd=ROOT)
    if rc:
        return rc
    print("[v2run] TOTAL %.0fs" % (time.time() - t_all), file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
