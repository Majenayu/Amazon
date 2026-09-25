#!/usr/bin/env python3
"""Parallel full-test blocking: ONE pass over S2/S3, sharded S1 index in RAM.

Each worker handles a shard of test-S1 rows (~215k rows each on 8 shards):
loads ONLY its shard into RAM as compact tuples, then streams the full
S2+S3 files once, emitting candidate pairs with features.

Keys: token (len>=4) + PIN only. No prefix keys (they caused 60k collisions).
Quality floor: pair needs >=1 shared name token or shared PIN.
"""
from __future__ import annotations
import multiprocessing as mp
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))

NON_ALNUM = re.compile(r"[^0-9a-z\u0900-\u097f ]+")
PIN = re.compile(r"(?<!\d)(\d{5,6})(?!\d)")
LEGAL = frozenset("inc incorporated llc ltd limited corp corporation co company plc gmbh pvt pte private liability partners llp srl sarl sa bv nv kg ag oy ab as pty holdings holding group".split())


def norm_name(s):
    if not s:
        return ""
    toks = NON_ALNUM.sub(" ", s.lower()).split()
    while toks and toks[-1] in LEGAL:
        toks.pop()
    return " ".join(toks)


def pins_of(a):
    if not a:
        return ()
    seen = []
    for m in PIN.findall(a):
        if m not in seen:
            seen.append(m)
        if len(seen) == 3:
            break
    return tuple(seen)


def parse(line):
    p = line.rstrip("\n").split("\t")
    while len(p) < 4:
        p.append("")
    n = norm_name(p[1])
    c = (p[3] or "").strip().lower()
    keys = []
    for t in n.split()[:6]:
        if len(t) >= 4:
            keys.append("T|" + c + "|" + t)
    for z in pins_of(p[2]):
        keys.append("Z|" + c + "|" + z)
    ntok = frozenset(n.split()) if n else frozenset()
    atok = frozenset((p[2] or "").lower().split())
    return (p[0], n, ntok, atok, c, pins_of(p[2]), keys)


def jac(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    i = len(a & b)
    return i / float(len(a) + len(b) - i) if i else 0.0


def triset(s):
    s = s.replace(" ", "_")
    if len(s) < 3:
        return {s} if s else set()
    return {s[i:i + 3] for i in range(len(s) - 2)}


def worker(args):
    shard_path, s2, s3, out_path, max_cand, collect = args
    t0 = time.time()
    recs = []
    index = {}
    with open(shard_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            r = parse(line)
            i = len(recs)
            recs.append(r)
            for k in r[6]:
                index.setdefault(k, []).append(i)
    n = len(recs)
    from array import array
    cnt = array("I", bytes(4 * n))
    has = bytearray(n)
    kept = cpos = scanned = 0
    with open(out_path, "w", encoding="utf-8") as out:
        out.write("s1_id\tother_id\tname_exact\tname_jacc\tname_tri\tname_pre3\taddr_jacc\tpin_eq\tcountry_eq\tlen_ratio\tlabel\n")
        for path in (s2, s3):
            with open(path, encoding="utf-8", errors="replace") as fh:
                next(fh, None)
                for line in fh:
                    scanned += 1
                    o = parse(line)
                    hits = []
                    for k in o[6]:
                        lst = index.get(k)
                        if lst:
                            hits.append((len(lst), lst))
                    if not hits:
                        continue
                    hits.sort(key=lambda x: x[0])
                    cand = []
                    seen = set()
                    for _, lst in hits:
                        for i in lst:
                            if i in seen:
                                continue
                            seen.add(i)
                            cand.append(i)
                            if len(cand) >= collect:
                                break
                        if len(cand) >= collect:
                            break
                    if len(cand) > max_cand:
                        ot = o[2]
                        cand.sort(key=lambda i: len(recs[i][2] & ot), reverse=True)
                        cand = cand[:max_cand]
                    op = set(o[5])
                    rows = []
                    for i in cand:
                        if cnt[i] >= max_cand:
                            continue
                        r = recs[i]
                        if not (r[2] & o[2]):
                            rp = set(r[5])
                            if not (op and rp and (op & rp)):
                                continue
                        cnt[i] += 1
                        has[i] = 1
                        t1, t2 = r[2], o[2]
                        g1, g2 = triset(r[1]), triset(o[1])
                        pp = 1.0 if (op and set(r[5]) and (op & set(r[5]))) else 0.0
                        l1, l2 = len(r[1]), len(o[1])
                        mx = l1 if l1 > l2 else l2
                        lr = (l1 if l1 < l2 else l2) / float(mx) if mx else 1.0
                        rows.append("%s\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t-1" % (
                            r[0], o[0],
                            1.0 if (r[1] == o[1] and r[1]) else 0.0,
                            jac(t1, t2), jac(g1, g2),
                            1.0 if (l1 >= 3 and r[1][:3] == o[1][:3]) else 0.0,
                            jac(r[3], o[3]), pp,
                            1.0 if r[4] == o[4] else 0.0, lr))
                        kept += 1
                    if rows:
                        out.write("\n".join(rows) + "\n")
    cov = sum(has)
    return (out_path, kept, scanned, cov, n, time.time() - t0)
