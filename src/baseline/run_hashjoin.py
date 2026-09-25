#!/usr/bin/env python3
"""Full run: hash-join block train, train model, hash-join block test, score."""
from __future__ import annotations
import argparse, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))
ROOT = os.path.dirname(os.path.dirname(HERE))
from baseline.hashjoin import HDR, load_slice, log, read_gt
from baseline.hashjoin2 import block_slice


def run_block(split, data_dir, out_path, chunk=250_000, tok_top=8):
    """Append-only + resumable: never truncates existing pair files.

    BUG GUARD: this used to open out_path with mode "w", which silently wiped a
    finished 1.7 GB train file when a test-only run was started. Now we always
    write to a part file and skip work whose part file already exists.
    """
    sub = os.path.join(data_dir, "train" if split == "train" else "test")
    s1_path = os.path.join(sub, "%s_source1.tsv" % split)
    total = sum(1 for _ in open(s1_path, encoding="utf-8", errors="replace")) - 1
    log("%s: %d S1 rows, chunk=%d" % (split, total, chunk))
    stats = {"pairs": 0, "pos": 0, "rare": 0}
    t0 = time.time()
    part = os.path.join(os.path.dirname(out_path), "_%s_parts" % split)
    os.makedirs(part, exist_ok=True)
    parts = []
    off = 0
    while off < total:
        pf = os.path.join(part, "p_%09d.tsv" % off)
        parts.append(pf)
        if os.path.exists(pf) and os.path.getsize(pf) > 0:
            log("  skip off=%d (part exists)" % off)
            off += chunk
            continue
        recs = load_slice(s1_path, off, chunk)
        if not recs:
            break
        gt = None
        if split == "train":
            gt = read_gt(os.path.join(data_dir, "train", "train_ground_truth.tsv"),
                         set(recs))
        with open(pf, "w", encoding="utf-8") as out:
            out.write(HDR + "\n")
            block_slice(split, sub, recs, gt, out, stats, tok_top=tok_top)
        off += len(recs)
        log("  off=%d pairs=%d pos=%d %.0fs" % (off, stats["pairs"], stats["pos"],
                                                time.time() - t0))
    # merge parts into out_path (header once)
    with open(out_path, "w", encoding="utf-8") as fout:
        fout.write(HDR + "\n")
        for pf in parts:
            if not os.path.exists(pf):
                continue
            with open(pf, encoding="utf-8", errors="replace") as fin:
                next(fin, None)
                for line in fin:
                    fout.write(line)
    log("%s DONE: %d pairs in %.0fs -> %s" % (split, stats["pairs"],
                                              time.time() - t0, out_path))
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=250_000)
    ap.add_argument("--tok-top", type=int, default=8)
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--test-only", action="store_true",
                    help="skip train blocking + training; reuse models/hj.joblib")
    a = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "data", "cache"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
    t_all = time.time()
    train_pairs = os.path.join(ROOT, "data", "cache", "hj_train.tsv")
    model = os.path.join(ROOT, "models", "hj")
    if not a.test_only:
        run_block("train", os.path.join(ROOT, "data"), train_pairs, a.chunk, a.tok_top)
        rc = subprocess.call([sys.executable, os.path.join(ROOT, "src", "baseline", "train.py"),
                              "--pairs", train_pairs,
                              "--s1", os.path.join(ROOT, "data", "train", "train_source1.tsv"),
                              "--gt", os.path.join(ROOT, "data", "train", "train_ground_truth.tsv"),
                              "--model-out", model,
                              "--max-iter", "100"], cwd=ROOT)
        if rc or a.train_only:
            return rc
    test_pairs = os.path.join(ROOT, "data", "cache", "hj_test.tsv")
    run_block("test", os.path.join(ROOT, "data"), test_pairs, a.chunk, a.tok_top)
    rc = subprocess.call([sys.executable, os.path.join(ROOT, "src", "baseline", "predict.py"),
                          "--pairs", test_pairs,
                          "--model", model + ".joblib",
                          "--test-dir", os.path.join(ROOT, "data", "test"),
                          "--matching-out", os.path.join(ROOT, "output", "matching_results.tsv"),
                          "--candidate-out", os.path.join(ROOT, "output", "candidate_pairs.tsv")],
                         cwd=ROOT)
    if rc:
        return rc
    log("TOTAL %.0fs" % (time.time() - t_all))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
