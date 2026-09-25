#!/usr/bin/env python3
"""Turn a --score-only probe into a recall-vs-volume curve.

Blocking has exactly one job: never lose a true match.  This script answers
"how much noise do I have to accept to keep X% of the true matches?", which is
the single number that caps the competition score.

    python src/baseline/analyze_score.py --scores data/cache/v3_score.tsv \
        --gt data/train/train_ground_truth.tsv --s1-limit 20000 --limit 150000
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))

from baseline.probe_recall import ids_in_window, load_gt, s1_ids_in_window  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--s1-limit", type=int, default=20000)
    ap.add_argument("--limit", type=int, default=150000)
    ap.add_argument("--other-rows", type=int, default=300000,
                    help="total source-2/3 rows scanned (2 x --limit)")
    args = ap.parse_args()

    base = args.split
    gt = load_gt(args.gt)
    keep = s1_ids_in_window("data/%s/%s_source1.tsv" % (base, base), args.s1_limit)
    window = (ids_in_window("data/%s/%s_source2.tsv" % (base, base), args.limit)
              | ids_in_window("data/%s/%s_source3.tsv" % (base, base), args.limit))

    denom = set()
    for s in keep:
        for o in gt.get(s, ()):
            if o in window:
                denom.add((s, o))

    # (s1, other) -> best evidence score seen
    best = {}
    n_lines = 0
    with open(args.scores, encoding="utf-8", errors="replace") as fh:
        next(fh, None)
        for line in fh:
            n_lines += 1
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            k = (p[0], p[1])
            sc = float(p[2])
            if sc > best.get(k, -1e9):
                best[k] = sc

    found = sum(1 for k in denom if k in best)
    print("probe pairs (distinct) : %s" % format(len(best), ","))
    print("probe lines            : %s" % format(n_lines, ","))
    print("true matches in window : %s" % format(len(denom), ","))
    print("true matches captured  : %s" % format(found, ","))
    if denom:
        print("CEILING recall (gate=0): %.4f" % (found / len(denom)))
    print("")
    print("%6s %9s %9s %9s %11s" % ("gate", "recall", "lost", "pairs", "pairs/other"))

    scores = sorted({v for v in best.values()})
    for gate in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]:
        keep_pairs = [k for k, v in best.items() if v >= gate]
        hit = sum(1 for k in denom if k in best and best[k] >= gate)
        r = hit / len(denom) if denom else 0.0
        print("%6d %9.4f %9d %9d %11.2f"
              % (gate, r, len(denom) - hit, len(keep_pairs),
                 len(keep_pairs) / max(1, args.other_rows)))

    # where do the misses come from?
    miss = [k for k in denom if k not in best]
    print("")
    print("misses at gate=0: %s" % format(len(miss), ","))
    by_other = Counter(o for _, o in miss)
    print("  distinct source-2/3 ids missed: %s" % format(len(by_other), ","))


if __name__ == "__main__":
    main()