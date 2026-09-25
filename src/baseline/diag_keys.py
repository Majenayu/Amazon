#!/usr/bin/env python3
"""Diagnostic: what do TRUE matches actually share, and what does it cost?

Builds the blocking key index on the first --n Source-1 rows with NO span cap,
then answers the two questions that decide the blocking design:

  1. RECALL: for sampled true matches, is there a shared key whose span is
     small enough to survive a span cap? (reported for caps 10..all)
  2. COST: how many candidates does one Source-2 record produce under each cap?

Usage:
    python src/baseline/diag_keys.py --data-dir data --n 100000
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))

from baseline.textnorm import make_record  # noqa: E402

CAPS = (10, 25, 50, 100, 200, 400, 1000, 10 ** 9)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--links", type=int, default=20000)
    args = ap.parse_args()

    train = os.path.join(args.data_dir, "train")
    s1_path = os.path.join(train, "train_source1.tsv")
    gt_path = os.path.join(train, "train_ground_truth.tsv")

    # ---------------------------------------------------------------- S1 index
    t0 = time.time()
    s1 = load_records(s1_path, limit=args.n)
    index = {}
    for sid, r in s1.items():
        for k in r["keys"]:
            index.setdefault(k, []).append(sid)
    spans = sorted(len(v) for v in index.values())
    log("S1 rows {:,} | keys {:,} | span p50={} p90={} p99={} max={} | {:.1f}s".format(
        len(s1), len(index), spans[len(spans) // 2], spans[int(len(spans) * .9)],
        spans[int(len(spans) * .99)], spans[-1], time.time() - t0))

    # ------------------------------------------------------------ sample links
    links, seen = [], set()
    with open(gt_path, encoding="utf-8", errors="replace") as fh:
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
    log("sampled {:,} true links".format(len(links)))

    others = {}
    for src in ("train_source2.tsv", "train_source3.tsv"):
        others.update(load_records(os.path.join(train, src), keep=seen))
        log("fetched {:,} matched records from {}".format(len(others), src))
        if len(others) == len(seen):
            break

    # ------------------------------------------------- recall per span cap
    print("\n--- RECALL: true links kept, by S1 key-span cap ---")
    n_ent = len({s for s, _ in links})
    for cap in CAPS:
        ok, kept = 0, set()
        for sid, oid in links:
            o = others.get(oid)
            if o is None:
                continue
            if any(k in index and sid in index[k] and len(index[k]) <= cap
                   for k in o["keys"]):
                ok += 1
                kept.add(sid)
        label = "all" if cap > 10 ** 8 else str(cap)
        print("%-8s links kept %-7s | entities with >=1 link %.2f%%" % (
            label, "%.2f%%" % (100.0 * ok / len(links)), 100.0 * len(kept) / n_ent))

    # ------------------------------------------------- cost per cap
    print("\n--- COST: candidates per Source-2 record, by span cap ---")
    sample = list(load_records(os.path.join(train, "train_source2.tsv"),
                               limit=200000).values())
    for cap in CAPS:
        tot = over = 0
        for o in sample:
            c = set()
            for k in o["keys"]:
                lst = index.get(k)
                if lst and len(lst) <= cap:
                    c.update(lst)
            tot += len(c)
            over += (len(c) > 10)
        label = "all" if cap > 10 ** 8 else str(cap)
        print("%-8s avg candidates/rec %8.1f | records with >10 cand %.1f%%" % (
            label, tot / len(sample), 100.0 * over / len(sample)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
