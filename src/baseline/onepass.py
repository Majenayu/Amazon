#!/usr/bin/env python3
"""ONE-PASS global blocker: index all Source-1 keys once, stream S2/S3 once.

Why it is fast
--------------
The slice-based approach scanned all 10M S2/S3 rows once per S1 slice
(7 slices -> 70M row scans). Here we build the Source-1 key index once and
scan S2/S3 exactly ONCE (10M row scans), which is ~7x less work for the
same candidate logic.

Candidate logic (identical intent to hashjoin2, better ordering)
----------------------------------------------------------------
1. keys: ('E', country, normalised name), ('T', country, token len>=4),
   ('Z', country, pin). Tokens covering more than --tok-cap Source-1 rows are
   dropped (they are pure noise, e.g. "services").
2. for each S2/S3 row, gather candidate S1 rows from its keys, score each by
   how many name tokens it shares, keep the best --collect, then accept at most
   --budget per (S1 row, source file).
3. features come from baseline.textnorm.features - the same 8 the model was
   trained on.

Memory: Source-1 is stored compactly (id / norm name / address text / country /
pins). No per-pair state is kept, so RAM stays around 1 GB on the full test set.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from array import array
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))

from baseline.textnorm import (  # noqa: E402
    FEATURE_NAMES, features, make_record, normalize_name, pins)

HDR = "s1_id\tother_id\t" + "\t".join(FEATURE_NAMES) + "\tlabel"


def log(m):
    print("[onepass] %s" % m, file=sys.stderr, flush=True)


def country_of(c):
    return (c or "").strip().lower()


def load_s1(path, limit=0):
    """Compact Source-1 store + key index."""
    recs = []
    index = defaultdict(list)
    with open(path, encoding="utf-8", errors="replace") as fh:
        next(fh, None)
        for line in fh:
            if limit and len(recs) >= limit:
                break
            p = line.rstrip("\n").split("\t")
            while len(p) < 4:
                p.append("")
            i = len(recs)
            n = normalize_name(p[1])
            c = country_of(p[3])
            pn = pins(p[2])
            # compact record: id, norm name, address text, country, pins
            recs.append((p[0], n, (p[2] or "").lower(), c, pn))
            if n:
                index[("E", c, n)].append(i)
            seen = set()
            for t in n.split():
                if len(t) >= 4 and t not in seen:
                    seen.add(t)
                    index[("T", c, t)].append(i)
            for z in pn:
                index[("Z", c, z)].append(i)
    return recs, index


def trim_index(index, tok_cap):
    """Drop token keys that cover too many Source-1 rows."""
    kept = {k: v for k, v in index.items()
            if k[0] != "T" or len(v) <= tok_cap}
    return kept


def read_gt(path, keep):
    gt = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        next(fh, None)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if keep is not None and p[0] not in keep:
                continue
            gt[p[0]] = frozenset(x for x in p[1].split(",") if x) if len(p) > 1 and p[1] else frozenset()
    return gt


def run(split, data_dir, out_path, tok_cap=200, collect=40, budget=25,
        s1_limit=0, gt_path=None):
    sub = os.path.join(data_dir, "train" if split == "train" else "test")
    s1_path = os.path.join(sub, "%s_source1.tsv" % split)
    srcs = [os.path.join(sub, "%s_source%d.tsv" % (split, i)) for i in (2, 3)]

    t0 = time.time()
    recs, index = load_s1(s1_path, s1_limit)
    log("S1 loaded: {:,} rows, {:,} index keys in {:.0f}s".format(
        len(recs), len(index), time.time() - t0))
    index = trim_index(index, tok_cap)
    log("index after tok_cap<=%d: {:,} keys".format(len(index)) % tok_cap)

    gt = None
    if split == "train" and gt_path and os.path.exists(gt_path):
        gt = read_gt(gt_path, None)
        log("ground truth loaded: {:,} entities".format(len(gt)))

    counters = array("I", bytes(4 * len(recs)))
    covered = bytearray(len(recs))
    stats = {"pairs": 0, "pos": 0, "scanned": 0}
    t1 = time.time()
    with open(out_path, "w", encoding="utf-8") as out:
        out.write(HDR + "\n")
        for si, src in enumerate(srcs):
            with open(src, encoding="utf-8", errors="replace") as fh:
                next(fh, None)
                for line in fh:
                    p = line.rstrip("\n").split("\t")
                    while len(p) < 4:
                        p.append("")
                    oid, nm, ad, co = p[0], p[1], p[2], p[3]
                    c = country_of(co)
                    n = normalize_name(nm)
                    stats["scanned"] += 1

                    cand = {}
                    if n:
                        for i in index.get(("E", c, n), ()):
                            cand[i] = cand.get(i, 0) + 3
                    for t in set(n.split()):
                        if len(t) >= 4:
                            for i in index.get(("T", c, t), ()):
                                cand[i] = cand.get(i, 0) + 1
                    for z in pins(ad):
                        for i in index.get(("Z", c, z), ()):
                            cand[i] = cand.get(i, 0) + 1
                    if not cand:
                        continue

                    o = make_record(oid, nm, ad, co)
                    otok = o["ntok"]
                    opins = set(o.get("pins") or ())

                    if len(cand) > collect:
                        ranked = sorted(cand, key=lambda i: (cand[i], -i),
                                        reverse=True)[:collect]
                    else:
                        ranked = sorted(cand, key=lambda i: cand[i], reverse=True)

                    for i in ranked:
                        if counters[i] >= budget:
                            continue
                        r = recs[i]
                        rn = r[1]
                        rpins = set(r[4] or ())
                        if not (set(rn.split()) & otok):
                            if not (rpins and opins and (rpins & opins)):
                                continue
                        counters[i] += 1
                        covered[i] = 1
                        rrec = {"id": r[0], "n": rn, "ntok": set(rn.split()),
                                "atok": set(r[2].split()), "country": r[3],
                                "pins": r[4]}
                        f = features(rrec, o)
                        lab = "-1"
                        if gt is not None:
                            lab = "1" if oid in gt.get(r[0], ()) else "0"
                        out.write(r[0] + "\t" + oid + "\t"
                                  + "\t".join("%.4f" % x for x in f)
                                  + "\t" + lab + "\n")
                        stats["pairs"] += 1
                        if lab == "1":
                            stats["pos"] += 1
                    if stats["scanned"] % 1000000 == 0:
                        log("  scanned {:,} | pairs {:,} | {:.0f}s".format(
                            stats["scanned"], stats["pairs"], time.time() - t1))
            log("  source%d done: scanned {:,} | pairs {:,} | {:.0f}s".format(
                si + 2, stats["scanned"], stats["pairs"], time.time() - t1))

    cov = sum(covered)
    log("DONE %s: {:,} pairs ({:,} true) in {:.0f}s".format(
        split, stats["pairs"], stats["pos"], time.time() - t0))
    log("  S1 rows with >=1 candidate: {:,}/{:,} ({:.1f}%)".format(
        cov, len(recs), 100.0 * cov / max(1, len(recs))))
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="One-pass global blocker.")
    ap.add_argument("--split", choices=("train", "test"), required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tok-cap", type=int, default=200)
    ap.add_argument("--collect", type=int, default=40)
    ap.add_argument("--budget", type=int, default=25)
    ap.add_argument("--s1-limit", type=int, default=0)
    ap.add_argument("--gt", default="")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(HERE))
    data_dir = a.data_dir if os.path.isabs(a.data_dir) else os.path.join(root, a.data_dir)
    out = a.out if os.path.isabs(a.out) else os.path.join(root, a.out)
    gt = a.gt if a.gt else os.path.join(data_dir, "train", "train_ground_truth.tsv")
    if not os.path.isabs(gt):
        gt = os.path.join(root, gt)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    run(a.split, data_dir, out, a.tok_cap, a.collect, a.budget, a.s1_limit, gt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())