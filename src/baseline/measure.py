"""Measure the two numbers that decide everything: blocking recall and the score.

Blocking recall = (true matches the blocker found) / (true matches that exist in
the scanned window). It is the HARD CEILING on the competition score: a true
match that never reaches the matcher can never be awarded, no matter how good
the classifier is.

A full pass over 10.3M Source-2/3 rows costs 1-2 hours, so recall is measured on
a slice: only ground-truth pairs whose Source-2/3 id falls inside the scanned
window are counted. That is an unbiased estimate of full-data recall and costs
minutes, not hours.

    # 1. make a probe (see block.py --score-only), minutes
    python src/baseline/block.py --split train --score-only --gt data/train/train_ground_truth.tsv --s1-limit 20000 --limit 150000 --out data/cache/probe.tsv

    # 2. recall + the recall/volume trade-off
    python src/baseline/measure.py recall --pairs data/cache/probe.tsv --gt data/train/train_ground_truth.tsv --s1-limit 20000 --limit 150000
    python src/baseline/measure.py curve   --pairs data/cache/probe.tsv --gt data/train/train_ground_truth.tsv --s1-limit 20000 --limit 150000

`curve` answers the only question that matters when widening the blocker: how
much noise must be accepted to keep X% of the true matches?
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))


def ids_in_window(path, n):
    """First n ids of a Source file (header skipped)."""
    out = set()
    with open(path, encoding="utf-8", errors="replace") as fh:
        next(fh, None)
        for i, line in enumerate(fh):
            if i >= n:
                break
            out.add(line.split("\t", 1)[0])
    return out


def load_gt(path):
    gt = defaultdict(set)
    with open(path, encoding="utf-8", errors="replace") as fh:
        next(fh, None)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) > 1 and p[1]:
                for x in p[1].split(","):
                    if x:
                        gt[p[0]].add(x)
    return gt


def known_true_pairs(gt, split, s1_limit, limit):
    """Ground-truth (s1, other) pairs that lie inside the scanned window."""
    base = "data/%s/%s_" % (split, split)
    keep = ids_in_window(base + "source1.tsv", s1_limit)
    window = ids_in_window(base + "source2.tsv", limit) | ids_in_window(base + "source3.tsv", limit)
    denom = set()
    for s in keep:
        for o in gt.get(s, ()):
            if o in window:
                denom.add((s, o))
    return denom, len(keep), len(window)


def load_pairs(path):
    """(s1, other) -> best evidence score."""
    best = {}
    n = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        next(fh, None)
        for line in fh:
            n += 1
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            k = (p[0], p[1])
            sc = float(p[2])
            if sc > best.get(k, -1e9):
                best[k] = sc
    return best, n


def cmd_recall(args):
    gt = load_gt(args.gt)
    denom, n_s1, n_win = known_true_pairs(gt, args.split, args.s1_limit, args.limit)
    best, n_lines = load_pairs(args.pairs)
    found = len(denom & set(best))
    print("pairs file lines   : %s" % format(n_lines, ","))
    print("distinct pairs     : %s" % format(len(best), ","))
    print("source-1 slice     : %s" % format(n_s1, ","))
    print("scanned id window  : %s" % format(n_win, ","))
    print("true matches known : %s" % format(len(denom), ","))
    print("true matches found : %s" % format(found, ","))
    if denom:
        print("")
        print("BLOCKING RECALL    : %.4f" % (found / len(denom)))
        missed = denom - set(best)
        print("missed             : %s" % format(len(missed), ","))
        for s, o in sorted(missed)[:8]:
            print("   MISS %s %s" % (s, o))
    return 0


def cmd_curve(args):
    gt = load_gt(args.gt)
    denom, _, _ = known_true_pairs(gt, args.split, args.s1_limit, args.limit)
    best, _ = load_pairs(args.pairs)
    found = sum(1 for k in denom if k in best)
    print("true matches in window : %s" % format(len(denom), ","))
    print("true matches captured  : %s" % format(found, ","))
    if denom:
        print("CEILING recall (gate=0): %.4f" % (found / len(denom)))
    print("")
    print("%6s %9s %9s %9s %11s" % ("gate", "recall", "lost", "pairs", "pairs/other"))
    for gate in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]:
        keep = [k for k, v in best.items() if v >= gate]
        hit = sum(1 for k in denom if k in best and best[k] >= gate)
        r = hit / len(denom) if denom else 0.0
        print("%6d %9.4f %9d %9d %11.2f"
              % (gate, r, len(denom) - hit, len(keep), len(keep) / max(1, n_other(args))))
    return 0


def n_other(args):
    return args.s1_limit * 2


def main():
    ap = argparse.ArgumentParser(description="blocking recall + score diagnostics")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, fn in (("recall", cmd_recall), ("curve", cmd_curve)):
        p = sub.add_parser(name)
        p.add_argument("--pairs", required=True, help="probe from block.py --score-only")
        p.add_argument("--gt", required=True)
        p.add_argument("--split", default="train")
        p.add_argument("--s1-limit", type=int, default=20000)
        p.add_argument("--limit", type=int, default=150000)
        p.set_defaults(fn=fn)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
