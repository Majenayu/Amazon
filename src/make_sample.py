#!/usr/bin/env python3
"""Build a small, LABEL-PRESERVING sample of the Amazon ML Challenge dataset.

Why this exists
---------------
The full dataset is ~2.5 GB / 26 million rows. No AI assistant can read that, and
loading it whole into RAM is slow and risky. This script cuts it down to a few MB
that you can open, paste, and iterate on in seconds.

The sample keeps the ground truth usable:

* S1 entities are chosen from ``train_ground_truth.tsv``, so every sampled S1 row
  still has its true match list.
* Every S2/S3 record that matches a chosen S1 is copied over (so labels stay valid).
* Extra random S2/S3 records are added as distractors, so the sample still looks
  like a real blocking problem.

Output layout mirrors the real one, so any script written against the sample works
on the full data by only changing the base directory:

    data/sample/train/train_source1.tsv   (etc.)
    data/sample/test/test_source1.tsv     (etc.)

Usage
-----
    python src/make_sample.py                  # defaults (~10-15 MB total)
    python src/make_sample.py --n-s1 2000      # tiny, pasteable into a chat prompt
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

DELIM = "\t"


# --------------------------------------------------------------------------- IO helpers
def read_header(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.readline()


def count_rows(path: str) -> int:
    """Count data rows (header excluded) without decoding the file."""
    n = 0
    with open(path, "rb") as fh:
        next(fh, None)
        for _ in fh:
            n += 1
    return n


def stride_for(total: int, target: int) -> int:
    return max(1, total // max(1, target))


def sample_plain(src_path: str, out_path: str, target: int) -> int:
    """Uniform 1/target stride sample; used for the unlabelled test files."""
    total = count_rows(src_path)
    step = stride_for(total, target)
    kept = 0
    with open(src_path, encoding="utf-8", errors="replace") as fin, \
            open(out_path, "w", encoding="utf-8", newline="") as fout:
        fout.write(fin.readline())
        for i, line in enumerate(fin):
            if i % step == 0:
                fout.write(line)
                kept += 1
    return kept


def sample_only_ids(src_path: str, out_path: str, keep_ids: set) -> int:
    """Copy only the rows whose id is in ``keep_ids``."""
    kept = 0
    with open(src_path, encoding="utf-8", errors="replace") as fin, \
            open(out_path, "w", encoding="utf-8", newline="") as fout:
        fout.write(fin.readline())
        for line in fin:
            if line.split(DELIM, 1)[0] in keep_ids:
                fout.write(line)
                kept += 1
    return kept


# ------------------------------------------------------------------ train: labels + S1
def sample_ground_truth(gt_path: str, out_path: str, n_s1: int):
    """Pick ~n_s1 Source-1 entities from the ground truth file.

    Returns (chosen_s1_ids, matched_s2s3_ids, written_rows).
    """
    total = count_rows(gt_path)
    step = stride_for(total, n_s1)

    chosen_s1 = set()
    matched = set()
    written = 0

    with open(gt_path, encoding="utf-8", errors="replace") as fin, \
            open(out_path, "w", encoding="utf-8", newline="") as fout:
        header = fin.readline()
        if not header:
            raise SystemExit("empty file: %s" % gt_path)
        fout.write(header)
        for i, line in enumerate(fin):
            if i % step:
                continue
            payload = line.rstrip("\n").split(DELIM)
            s1_id = payload[0].strip()
            if not s1_id:
                continue
            chosen_s1.add(s1_id)
            if len(payload) > 1 and payload[1].strip():
                matched.update(x for x in payload[1].strip().split(",") if x)
            fout.write(line)
            written += 1

    if written == 0:
        raise SystemExit("no ground-truth rows sampled from %s" % gt_path)
    return chosen_s1, matched, written


def sample_source(src_path: str, out_path: str, keep_ids: set, n_decoy: int):
    """Copy every row whose id is in ``keep_ids`` (true matches), plus ~n_decoy
    random distractor rows so the sample stays a realistic blocking problem."""
    total = count_rows(src_path)
    step = stride_for(total, n_decoy)

    hits = decoys = 0
    with open(src_path, encoding="utf-8", errors="replace") as fin, \
            open(out_path, "w", encoding="utf-8", newline="") as fout:
        fout.write(fin.readline())
        for i, line in enumerate(fin):
            eid = line.split(DELIM, 1)[0]
            if eid in keep_ids:
                fout.write(line)
                hits += 1
            elif i % step == 0 and decoys < n_decoy:
                fout.write(line)
                decoys += 1
    return hits, decoys


# --------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="Build a small labelled sample of the dataset.")
    ap.add_argument("--n-s1", type=int, default=10_000,
                    help="sampled Source-1 entities with labels (default 10000)")
    ap.add_argument("--n-decoy", type=int, default=15_000,
                    help="random distractor rows per train source (default 15000)")
    ap.add_argument("--n-test-s1", type=int, default=10_000,
                    help="sampled test Source-1 rows (default 10000)")
    ap.add_argument("--n-test-decoy", type=int, default=15_000,
                    help="sampled rows per test source (default 15000)")
    args = ap.parse_args()

    train = os.path.join(DATA_DIR, "train")
    test = os.path.join(DATA_DIR, "test")
    out_train = os.path.join(DATA_DIR, "sample", "train")
    out_test = os.path.join(DATA_DIR, "sample", "test")
    os.makedirs(out_train, exist_ok=True)
    os.makedirs(out_test, exist_ok=True)

    required = [os.path.join(train, f) for f in (
        "train_ground_truth.tsv", "train_source1.tsv", "train_source2.tsv",
        "train_source3.tsv")] + [
        os.path.join(test, f) for f in
        ("test_source1.tsv", "test_source2.tsv", "test_source3.tsv")]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print("Missing data files (run from the repo root):", file=sys.stderr)
        for p in missing:
            print("  - " + p, file=sys.stderr)
        return 1

    print("Sampling ground truth + Source 1 ...")
    gt_out = os.path.join(out_train, "train_ground_truth.tsv")
    s1_ids, matched_ids, n_gt = sample_ground_truth(
        os.path.join(train, "train_ground_truth.tsv"), gt_out, args.n_s1)
    n_s1 = sample_only_ids(
        os.path.join(train, "train_source1.tsv"),
        os.path.join(out_train, "train_source1.tsv"), s1_ids)
    print("  %d labelled S1 entities (%d rows matched in source1) | %d distinct true-match ids"
          % (n_gt, n_s1, len(matched_ids)))

    for src in ("train_source2.tsv", "train_source3.tsv"):
        print("Sampling %s ..." % src)
        hits, decoys = sample_source(
            os.path.join(train, src), os.path.join(out_train, src),
            matched_ids, args.n_decoy)
        print("  %d true matches + %d distractors" % (hits, decoys))

    print("Sampling test files (no labels available) ...")
    for src, target in (("test_source1.tsv", args.n_test_s1),
                        ("test_source2.tsv", args.n_test_decoy),
                        ("test_source3.tsv", args.n_test_decoy)):
        kept = sample_plain(os.path.join(test, src), os.path.join(out_test, src), target)
        print("  %s: %d rows" % (src, kept))

    total = sum(os.path.getsize(os.path.join(d, f))
                for d in (out_train, out_test) for f in os.listdir(d))
    print("\nDone. Sample written to:\n  %s\n  %s" % (out_train, out_test))
    print("Total sample size: %.1f MB" % (total / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

