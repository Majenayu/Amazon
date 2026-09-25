#!/usr/bin/env python3
"""Diagnose v2 recall on a 20k S1 slice WITHOUT loading the pickle (low RAM)."""
from __future__ import annotations
import argparse, os, sys
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))
from baseline.textnorm import normalize_name, pins
from baseline.hashjoin import norm_country, read_gt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True)
    ap.add_argument("--s1", required=True)
    ap.add_argument("--gt", default="data/train/train_ground_truth.tsv")
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()

    recs, order = {}, []
    with open(a.s1, encoding="utf-8", errors="replace") as fh:
        next(fh, None)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            while len(p) < 4:
                p.append("")
            c = norm_country(p[3])
            n = normalize_name(p[1])
            toks = set(n.split())
            if len(order) >= a.n:
                break
            recs[p[0]] = (n, toks, set(pins(p[2])), c)
            order.append(p[0])
    print("S1 slice: %d" % len(order), flush=True)

    exact = defaultdict(list)
    posting = defaultdict(list)
    for sid, (n, toks, pn, c) in recs.items():
        if n:
            exact[(c, n)].append(sid)
        for t in toks:
            if len(t) >= 4:
                posting[(c, t)].append(sid)
    plen = {k: len(v) for k, v in posting.items()}
    gt = read_gt(a.gt, set(order))
    n_true = sum(len(v) for v in gt.values())
    print("GT links in slice: %d" % n_true, flush=True)

    for cap in (50, 150, 500):
        rare = {k for k, v in plen.items() if v <= cap}
        hit_links = hit_ent = 0
        ent_hit = set()
        with open(a.pre, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                q = line.rstrip("\n").split("\t")
                if len(q) < 6:
                    continue
                oid, on, oc = q[0], q[1], q[4]
                cand = set(exact.get((oc, on), ()))
                for t in (q[2].split(";") if q[2] else ()):
                    if len(t) >= 4 and (oc, t) in rare:
                        cand.update(posting[(oc, t)])
                        if len(cand) > 400:
                            break
                if not cand:
                    continue
                for sid in cand:
                    if oid in gt.get(sid, ()):
                        hit_links += 1
                        ent_hit.add(sid)
        print("cap=%d links=%d/%d (%.1f%%) ents=%d/%d (%.1f%%)" % (
            cap, hit_links, n_true, 100.0 * hit_links / max(n_true, 1),
            len(ent_hit), len(gt), 100.0 * len(ent_hit) / max(len(gt), 1)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
