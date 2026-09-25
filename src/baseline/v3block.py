#!/usr/bin/env python3
"""v3 blocker: ONE pass over Source-2/3, union of 5 key types, 20 features.

Improvements over v1/v2
-----------------------
1. UNION KEYS. v1/v2 used exact-name + rare-token + PIN. v3 adds
     S  sorted-token signature   -> survives word-order swaps
     X  per-token 4-char prefix  -> survives typos inside a word
   which is what lifts the blocking recall ceiling.
2. RANKED SELECTION. Candidates are ordered by a combined evidence score
   (shared tokens x2 + shared token-prefixes + PIN + signature + exact name),
   so a true match outranks generic noise instead of losing to file order.
3. WIDE BUDGET. --budget is per (S1 row, source file). A large budget plus the
   model's own threshold beats a tight budget: dropping a true match is
   unrecoverable, an extra candidate is only a little noise.

Memory: Source-1 is stored compactly (id / name / address text / country /
pins); the address token set is built only for rows that actually become
candidates.

Usage
-----
    python src/baseline/v3block.py --split train --out data/cache/v3_train.tsv
    python src/baseline/v3block.py --split test  --out data/cache/v3_test.tsv
"""

from __future__ import annotations

import argparse
import heapq
import os
import sys
import time
from array import array
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))

from baseline.textnorm import normalize_name, pins  # noqa: E402
from baseline.v3feat import FEATURE_NAMES_V3, features_v3  # noqa: E402

HDR = "s1_id\tother_id\t" + "\t".join(FEATURE_NAMES_V3) + "\tlabel"
SCORE_HDR = "s1_id\tother_id\tevidence\tlabel"
TOK_MIN = 4          # tokens shorter than this are too common to key on
PRE_MIN = 4          # token-prefix key length


def log(m):
    print("[v3] %s" % m, file=sys.stderr, flush=True)


def cc(c):
    return (c or "").strip().lower()


def key_sig(n):
    return "|".join(sorted(n.split()))


def load_s1(path, limit=0):
    """Compact Source-1 store + the union key index."""
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
            c = cc(p[3])
            pn = pins(p[2])
            recs.append((p[0], n, (p[2] or "").lower(), c, pn))
            if n:
                index["E|" + c + "|" + n].append(i)
                index["S|" + c + "|" + key_sig(n)].append(i)
            seen_t, seen_x = set(), set()
            for t in n.split():
                if len(t) >= TOK_MIN and t not in seen_t:
                    seen_t.add(t)
                    index["T|" + c + "|" + t].append(i)
                if len(t) >= PRE_MIN and t[:PRE_MIN] not in seen_x:
                    seen_x.add(t[:PRE_MIN])
                    index["X|" + c + "|" + t[:PRE_MIN]].append(i)
            for z in pn:
                index["Z|" + c + "|" + z].append(i)
    return recs, index


def trim(index, tok_cap, pre_cap):
    """Drop keys that cover too many Source-1 rows - they are pure noise."""
    return {k: v for k, v in index.items()
            if (k[0] == "T" and len(v) <= tok_cap)
            or (k[0] == "X" and len(v) <= pre_cap)
            or k[0] in ("E", "S", "Z")}


def keys_for(n, c, pn):
    """Return (weight, key) pairs.  Weight = how strong that key type is.

    E exact name and S sorted signature are near-certain matches, T/Z are solid
    evidence, X (4-char token prefix) is only a typo hint.  The weight both
    orders which posting lists we visit first and how we rank candidates.
    """
    out = []
    if n:
        out.append((8, "E|" + c + "|" + n))
        out.append((6, "S|" + c + "|" + key_sig(n)))
    seen_t, seen_x = set(), set()
    for t in n.split():
        if len(t) >= TOK_MIN and t not in seen_t:
            seen_t.add(t)
            out.append((3, "T|" + c + "|" + t))
        if len(t) >= PRE_MIN and t[:PRE_MIN] not in seen_x:
            seen_x.add(t[:PRE_MIN])
            out.append((1, "X|" + c + "|" + t[:PRE_MIN]))
    for z in pn:
        out.append((3, "Z|" + c + "|" + z))
    return out


def read_gt(path):
    """ground truth: source-1 id -> set of matched source-2/3 ids.

    The value column holds a COMMA-separated list of matches.
    """
    gt = defaultdict(set)
    with open(path, encoding="utf-8", errors="replace") as fh:
        next(fh, None)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2 and p[1]:
                for x in p[1].split(","):
                    if x:
                        gt[p[0]].add(x)
    return gt


def atok_of(cache, key, addr, limit=200000):
    s = cache.get(key)
    if s is None:
        s = set(addr.split())
        if len(cache) < limit:
            cache[key] = s
    return s


def run(args):
    base = args.split
    s1 = args.s1 or "data/%s/%s_source1.tsv" % (base, base)
    s2 = args.s2 or "data/%s/%s_source2.tsv" % (base, base)
    s3 = args.s3 or "data/%s/%s_source3.tsv" % (base, base)

    if args.gt:
        t0 = time.time()
        gt = read_gt(args.gt)
        log("GT loaded: %s source-1 ids in %.1fs"
            % (format(len(gt), ","), time.time() - t0))
    else:
        gt = None

    t0 = time.time()
    recs, index = load_s1(s1, args.s1_limit)
    log("S1 loaded: %s rows, %s keys in %.0fs"
        % (format(len(recs), ","), format(len(index), ","), time.time() - t0))
    index = trim(index, args.tok_cap, args.pre_cap)
    log("keys after trim (T<=%d X<=%d): %s"
        % (args.tok_cap, args.pre_cap, format(len(index), ",")))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pending = args.out + ".partial"
    total = accepted = positives = 0
    t0 = time.time()

    with open(pending, "w", encoding="utf-8", buffering=1 << 22) as fo:
        fo.write((SCORE_HDR if args.score_only else HDR) + "\n")
        for src, path in (("S2", s2), ("S3", s3)):
            counts = array("I", bytes(4 * len(recs)))
            cache = {}
            n_rows = 0
            with open(path, encoding="utf-8", errors="replace") as fh:
                next(fh, None)
                for line in fh:
                    n_rows += 1
                    if args.limit and n_rows > args.limit:
                        break
                    if n_rows % 200000 == 0:
                        log("%s: %s rows | %s pairs | %.0fs | %.0f rows/s"
                            % (src, format(n_rows, ","), format(total, ","),
                               time.time() - t0, n_rows / max(time.time() - t0, 1e-9)))
                    p = line.rstrip("\n").split("\t")
                    while len(p) < 4:
                        p.append("")
                    n2 = normalize_name(p[1])
                    if not n2:
                        continue
                    oid, c2 = p[0], cc(p[3])
                    atok2 = set((p[2] or "").lower().split())
                    pins2 = set(pins(p[2]))
                    t2 = set(n2.split())
                    pre2 = {t[:PRE_MIN] for t in t2 if len(t) >= PRE_MIN}

                    # Phase 1 - visit posting lists ordered by (key strength,
                    # selectivity) and stop at --post-budget, so one common word
                    # cannot turn a row into thousands of candidates.
                    lst_info = []
                    for w, k in keys_for(n2, c2, pins2):
                        lst = index.get(k)
                        if lst:
                            lst_info.append((w, len(lst), lst))
                    if not lst_info:
                        continue
                    lst_info.sort(key=lambda x: (-x[0], x[1]))
                    cand = {}
                    visited = 0
                    for w, df, lst in lst_info:
                        if visited and visited + df > args.post_budget:
                            continue
                        for i in lst:
                            cand[i] = cand.get(i, 0) + w
                        visited += df
                        if visited >= args.post_budget:
                            break
                    if not cand:
                        continue

                    # Phase 2 - the exact evidence score is only evaluated for
                    # the --pre-collect strongest candidates.  Scoring every
                    # candidate is the single biggest cost in the pipeline.
                    top = heapq.nlargest(args.pre_collect, cand.items(),
                                         key=lambda kv: kv[1])
                    ranked = []
                    for i, w in top:
                        n1 = recs[i][1]
                        t1 = set(n1.split())
                        pre1 = {t[:PRE_MIN] for t in t1 if len(t) >= PRE_MIN}
                        sc = 2 * len(t1 & t2) + len(pre1 & pre2)
                        if recs[i][4] and pins2 and (set(recs[i][4]) & pins2):
                            sc += 2
                        if t1 and t1 == t2:
                            sc += 6
                        if n1 == n2:
                            sc += 10
                        if sc >= args.gate:
                            ranked.append((sc, i))
                    if not ranked:
                        continue
                    ranked.sort(reverse=True)

                    r2 = {"n": n2, "atok": atok2, "country": c2, "pins": tuple(pins2)}
                    for sc, i in ranked[: args.collect]:
                        if sc < args.strong and counts[i] >= args.budget:
                            continue
                        counts[i] += 1
                        sid = recs[i][0]
                        if args.score_only:
                            # analysis mode: no features, just the evidence score
                            lab = ""
                            if gt is not None:
                                lab = "1" if oid in gt.get(sid, ()) else "0"
                                if lab == "1":
                                    positives += 1
                            fo.write("%s\t%s\t%g\t%s\n" % (sid, oid, sc, lab))
                            accepted += 1
                            continue
                        r1 = {"n": recs[i][1],
                              "atok": atok_of(cache, i, recs[i][2]),
                              "country": recs[i][3],
                              "pins": recs[i][4]}
                        fv = features_v3(r1, r2)
                        lab = ""
                        if gt is not None:
                            lab = "1" if oid in gt.get(sid, ()) else "0"
                            if lab == "1":
                                positives += 1
                        fo.write(sid + "\t" + oid + "\t" +
                                 "\t".join(("%.5g" % v) for v in fv) + "\t" + lab + "\n")
                        accepted += 1
                        if accepted % 4000000 == 0:
                            fo.flush()
                    total += 1

    el = time.time() - t0
    if os.path.exists(args.out):
        os.remove(args.out)
    os.replace(pending, args.out)
    log("DONE %s feature-rows in %.1fs" % (format(accepted, ","), el))
    log("output: %s (%.2f GB)" % (args.out, os.path.getsize(args.out) / 1e9))
    if positives:
        log("positives: %s (%.4f%% of pairs)"
            % (format(positives, ","), 100.0 * positives / max(accepted, 1)))


def main():
    ap = argparse.ArgumentParser(description="v3 one-pass union-key blocker")
    ap.add_argument("--split", default="test", choices=("train", "test"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--s1", default=None)
    ap.add_argument("--s2", default=None)
    ap.add_argument("--s3", default=None)
    ap.add_argument("--gt", default=None,
                    help="ground-truth tsv to label pairs (train split only)")
    ap.add_argument("--s1-limit", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tok-cap", type=int, default=300,
                    help="max Source-1 rows sharing a token key")
    ap.add_argument("--pre-cap", type=int, default=400,
                    help="max Source-1 rows sharing a token-prefix key")
    ap.add_argument("--gate", type=float, default=2.0,
                    help="min combined evidence score to become a candidate")
    ap.add_argument("--strong", type=float, default=6.0,
                    help="score that is never cut by the budget")
    ap.add_argument("--budget", type=int, default=15,
                    help="max weak candidates per (source-1 row, source file)")
    ap.add_argument("--collect", type=int, default=15,
                    help="max candidates written per source-2/3 row")
    ap.add_argument("--pre-collect", type=int, default=15,
                    help="candidates given the exact evidence score (phase 2)")
    ap.add_argument("--post-budget", type=int, default=150,
                    help="max posting-list entries visited per source-2/3 row")
    ap.add_argument("--score-only", action="store_true",
                    help="write s1/other/evidence/label only (fast recall analysis)")
    args = ap.parse_args()
    if args.out is None:
        args.out = "data/cache/v3_%s.tsv" % args.split
    run(args)


if __name__ == "__main__":
    main()