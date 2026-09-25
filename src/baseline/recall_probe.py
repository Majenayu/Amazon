#!/usr/bin/env python3
"""Recall probe on a random labelled sample: how far do exact/PIN/token keys go?"""
from __future__ import annotations
import os, random, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))
from baseline.textnorm import make_record

ROOT = os.path.dirname(os.path.dirname(HERE))
SD = os.path.join(ROOT, "data", "sample", "train")

def rec(p):
    p += [""] * (4 - len(p))
    return make_record(p[0], p[1], p[2], p[3])

t0 = time.time()
gt = {}
with open(os.path.join(SD, "train_ground_truth.tsv"), encoding="utf-8") as fh:
    next(fh, None)
    for line in fh:
        p = line.rstrip("\n").split("\t")
        gt[p[0]] = set(p[1].split(",")) if len(p) > 1 and p[1].strip() else set()
s1 = {}
with open(os.path.join(SD, "train_source1.tsv"), encoding="utf-8") as fh:
    next(fh, None)
    for line in fh:
        p = line.rstrip("\n").split("\t")
        s1[p[0]] = rec(p)
both = {}
for fn in ("train_source2.tsv", "train_source3.tsv"):
    with open(os.path.join(SD, fn), encoding="utf-8") as fh:
        next(fh, None)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            both[p[0]] = rec(p)
print("loaded %.0fs  gt=%d s1=%d other=%d" % (time.time() - t0, len(gt), len(s1), len(both)), flush=True)

random.seed(0)
samp = random.sample(list(gt.keys()), min(3000, len(gt)))
tot = ex = pin = tok = non = 0
for sid in samp:
    g = gt[sid]
    if not g:
        non += 1
        continue
    r = s1.get(sid)
    if r is None:
        tot += 1
        continue
    rn, rp, rc = r["n"], set(r.get("pins") or ()), r["country"]
    hit_e = hit_p = hit_t = False
    for oid in g:
        o = both.get(oid)
        if o is None:
            continue
        tot += 1
        if o["n"] and o["n"] == rn:
            ex += 1; hit_e = True
        if rp and set(o.get("pins") or ()) & rp:
            pin += 1; hit_p = True
        if r["ntok"] & o["ntok"]:
            tok += 1; hit_t = True
print("sampled=%d empty=%d links=%d" % (len(samp), non, tot), flush=True)
print("exact-name link recall : %.1f%%" % (100.0 * ex / tot if tot else 0), flush=True)
print("pin      link recall : %.1f%%" % (100.0 * pin / tot if tot else 0), flush=True)
print("token    link recall : %.1f%%" % (100.0 * tok / tot if tot else 0), flush=True)
