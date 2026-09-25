#!/usr/bin/env python3
"""Per-key-type breakdown: which blocking keys give recall, and which cost time.

block_keys() emits three kinds of keys:
    P|country|xxx   first 3 chars of the cleaned name   (1 per record, VERY common)
    T|country|tok   name tokens of length >= 4          (up to 6 per record)
    Z|country|pin   postal code                          (up to 3 per record)

This tells us which of them is worth keeping for the full 2.2M-row run.

Usage:  python src/baseline/diag_keytypes.py --data-dir data --n 100000
"""

from __future__ import annotations

import argparse
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))

from baseline.textnorm import make_record  # noqa: E402

TYPES = ("P", "T", "Z")


def log(m):
    print(m, file=sys.stderr, flush=True)


def load_records(path, limit=None, keep=None):
    out = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        next(fh, None)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if keep is not None and p[0] not in keep:
                continue
            while len(p) < 4:
                p.append("")
            out[p[0]] = make_record(p[0], p[1], p[2], p[3])
            if limit and len(out) >= limit:
                break
    return out


def split_keys(keys, t):
    return [k for k in keys if k.startswith(t + "|")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--links", type=int, default=20000)
    args = ap.parse_args()

    train = os.path.join(args.data_dir, "train")
    s1 = load_records(os.path.join(train, "train_source1.tsv"), limit=args.n)

    idx = {t: {} for t in TYPES}
    for sid, r in s1.items():
        for t in TYPES:
            for k in split_keys(r["keys"], t):
                idx[t].setdefault(k, []).append(sid)

    print("--- key inventory on {:,} Source-1 rows ---".format(len(s1)))
    for t in TYPES:
        spans = sorted(len(v) for v in idx[t].values())
        if not spans:
            print("%s: no keys" % t)
            continue
        print("%s: %7d keys | avg span %7.1f | p50 %4d p90 %5d p99 %6d max %6d" % (
            t, len(spans), sum(spans) / len(spans), spans[len(spans) // 2],
            spans[int(len(spans) * .9)], spans[int(len(spans) * .99)], spans[-1]))

    # -------------------------------------------------------------- true links
    links, seen = [], set()
    with open(os.path.join(train, "train_ground_truth.tsv"), encoding="utf-8",
              errors="replace") as fh:
        next(fh, None)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if p[0] not in s1:
                continue
            if len(p) > 1 and p[1]:
                for x in p[1].split(","):
                    if x and x not in seen:
                        links.append((p[0], x))
                        seen.add(x)
    random.Random(0).shuffle(links)
    links = links[:args.links]

    others = load_records(os.path.join(train, "train_source2.tsv"), keep=seen)
    others.update(load_records(os.path.join(train, "train_source3.tsv"), keep=seen))
    log("fetched {:,} matched records from {:,} links".format(len(others), len(links)))

    def best_span(sid, other, t):
        s1_keys = set(split_keys(s1[sid]["keys"], t))
        best = None
        for k in split_keys(other["keys"], t):
            lst = idx[t].get(k)
            if lst and sid in lst:
                n = len(lst)
                if best is None or n < best:
                    best = n
        return best

    print("\n--- RECALL: {:,} true links, by key type (min shared-key span) ---".format(len(links)))
    usable = [(s, o) for s, o in links if o in others]
    for t in TYPES:
        for cap in (100, 500, 2000, 10 ** 9):
            ok = sum(1 for s, o in usable
                     if (lambda b: b is not None and b <= cap)(best_span(s, others[o], t)))
            label = "all" if cap > 10 ** 8 else str(cap)
            print("  %s keys, span<=%-6s : %6.2f%% of links" % (t, label, 100.0 * ok / len(usable)))
    ok_any = sum(1 for s, o in usable
                 if any(best_span(s, others[o], t) is not None for t in ("T", "Z")))
    ok_either = sum(1 for s, o in usable
                    if any(best_span(s, others[o], t) is not None for t in TYPES))
    print("  T or Z (no prefix)   : %6.2f%%" % (100.0 * ok_any / len(usable)))
    print("  any key type         : %6.2f%%" % (100.0 * ok_either / len(usable)))

    # ------------------------------------------------------------------- cost
    print("\n--- COST: candidates per Source-2 record (100k-row index) ---")
    sample = list(load_records(os.path.join(train, "train_source2.tsv"),
                               limit=150000).values())
    for t in TYPES:
        for cap in (500, 10 ** 9):
            tot = 0
            for o in sample:
                c = set()
                for k in split_keys(o["keys"], t):
                    lst = idx[t].get(k)
                    if lst and len(lst) <= cap:
                        c.update(lst)
                tot += len(c)
            label = "all" if cap > 10 ** 8 else str(cap)
            print("  %s keys, span<=%-6s : %8.1f candidates/record" % (
                t, label, tot / len(sample)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
