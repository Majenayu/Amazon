#!/usr/bin/env python3
"""Measure BLOCKING RECALL against ground truth.

Recall = (true matches the blocker found) / (true matches that exist in the
data we scanned).  This is the hard ceiling on the competition score: if a true
match never reaches the matcher, no classifier can ever award it.

Because scanning all 20M Source-2/3 rows takes ~100 minutes, the script works
on a slice: it only counts ground-truth pairs whose Source-2/3 id is inside the
window that was actually scanned (--limit).  The resulting number is an
unbiased estimate of full-data recall.

    python src/baseline/probe_recall.py \
        --s1-limit 20000 --limit 300000 \
        --pairs data/cache/v3_probe.tsv --gt data/train/train_ground_truth.tsv
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))


def ids_in_window(path, n):
    """First n ids of a Source-2/3 file (header skipped)."""
    out = set()
    with open(path, encoding="utf-8", errors="replace") as fh:
        next(fh, None)
        for i, line in enumerate(fh):
            if i >= n:
                break
            out.add(line.split("\t", 1)[0])
    return out


def s1_ids_in_window(path, n):
    out = set()
    with open(path, encoding="utf-8", errors="replace") as fh:
        next(fh, None)
        for i, line in enumerate(fh):
            if i >= n:
                break
            out.add(line.split("\t", 1)[0])
    return out


def load_gt(path):
    """source-1 id -> set of matched source-2/3 ids (comma-separated column)."""
    gt = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        next(fh, None)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) > 1 and p[1]:
                for x in p[1].split(","):
                    if x:
                        gt.setdefault(p[0], set()).add(x)
    return gt


def load_pairs(path, only_pos=False):
    pairs = set()
    n = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        next(fh, None)
        for line in fh:
            n += 1
            if only_pos and not line.rstrip("\n").endswith("\t1"):
                continue
            p = line.split("\t", 2)
            pairs.add((p[0], p[1]))
    return pairs, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--s1-limit", type=int, default=20000)
    ap.add_argument("--limit", type=int, default=300000)
    ap.add_argument("--only-pos", action="store_true")
    args = ap.parse_args()

    base = args.split
    s1 = "data/%s/%s_source1.tsv" % (base, base)
    s2 = "data/%s/%s_source2.tsv" % (base, base)
    s3 = "data/%s/%s_source3.tsv" % (base, base)

    gt = load_gt(args.gt)
    keep_s1 = s1_ids_in_window(s1, args.s1_limit)
    window = ids_in_window(s2, args.limit) | ids_in_window(s3, args.limit)

    denom = set()
    for s in keep_s1:
        for o in gt.get(s, ()):
            if o in window:
                denom.add((s, o))

    pairs, n_lines = load_pairs(args.pairs, args.only_pos)
    numer = denom & pairs

    print("pairs file lines   : %s" % format(n_lines, ","))
    print("distinct pairs     : %s" % format(len(pairs), ","))
    print("source-1 slice     : %s" % format(len(keep_s1), ","))
    print("scanned id window  : %s" % format(len(window), ","))
    print("true matches known : %s" % format(len(denom), ","))
    print("true matches found : %s" % format(len(numer), ","))
    if denom:
        print("")
        print("BLOCKING RECALL    : %.4f" % (len(numer) / len(denom)))
        missed = denom - pairs
        print("missed              : %s" % format(len(missed), ","))
        for s, o in sorted(missed)[:8]:
            print("   MISS %s %s" % (s, o))


if __name__ == "__main__":
    main()