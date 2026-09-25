"""Candidate generation: ONE streaming pass over Source-2/3 against a union key
index built on Source-1.

Test has 1,732,544 x 9,969,589 = about 1.7e13 possible pairs, so this stage is
what makes the problem tractable - and its recall is the hard ceiling on the
final score.

How recall is bought
--------------------
1. UNION of six key types (exact name, sorted signature, token, 4-char token
   prefix, postal code, address token). A pair is a candidate if ANY key
   links it. Address keys are the only signal for transliteration pairs
   whose names share nothing.
2. KEY TRIMMING. Any token/prefix/address key shared by more than --tok-cap /
   --pre-cap / --addr-cap Source-1 rows is dropped; such keys are pure noise.
3. RANKED SELECTION. Candidates are ordered by a combined evidence score, so a
   true match outranks generic noise instead of losing on file order. An earlier
   first-come-first-served budget was measured to destroy recall.
4. WIDE BUDGET plus a model-side threshold. --budget bounds the file size; the
   classifier does the precision work. Dropping a true match is unrecoverable,
   one extra candidate is only a little noise.

CRITICAL: train and test must be blocked at the SAME density. The threshold is
tuned on the training candidate distribution, so if training uses --limit and
test does not, the tuned threshold is calibrated against an easier problem than
the one being served. Run 1 of this pipeline lost points to exactly that; see
docs/WORKFLOW.md.

Memory: Source-1 is stored compactly (id / name / address text / country /
pins). Address token sets are built only for rows that become candidates.

Usage
-----
    python src/baseline/block.py --split train --out data/cache/train_pairs.tsv \
        --gt data/train/train_ground_truth.tsv
    python src/baseline/block.py --split test  --out data/cache/test_pairs.tsv
    python src/baseline/block.py --split train --score-only --s1-limit 20000 \
        --limit 150000 --out data/cache/probe.tsv     # recall probe, minutes
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

from baseline.core import (  # noqa: E402
    HDR, PRE_MIN, SCORE_HDR, TOK_MIN, addr_tokens, cc, features, keys_for,
    key_sig, normalize_name, pins,
)


def log(m):
    print("[block] %s" % m, file=sys.stderr, flush=True)


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
            addr_raw = p[2] or ""
            atok = addr_tokens(addr_raw)
            recs.append((p[0], n, addr_raw.lower(), c, pn, atok))
            for _, k in keys_for(n, c, pn, atok):
                index[k].append(i)
    return recs, index


def trim(index, tok_cap, pre_cap, addr_cap=400):
    """Drop keys that cover too many Source-1 rows - they are pure noise.

    Address keys get their own cap: generic-but-unskipped words (e.g. market,
    chennai) legitimately cover more rows than a name token, but keys above
    the cap are still pure noise that would flood the candidate budget.
    """
    return {k: v for k, v in index.items()
            if (k[0] == "T" and len(v) <= tok_cap)
            or (k[0] == "X" and len(v) <= pre_cap)
            or (k[0] == "A" and len(v) <= addr_cap)
            or k[0] in ("E", "S", "Z")}


def read_gt(path):
    """source-1 id -> set of matched source-2/3 ids (comma-separated column)."""
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
    index = trim(index, args.tok_cap, args.pre_cap, args.addr_cap)
    log("keys after trim (T<=%d X<=%d A<=%d): %s"
        % (args.tok_cap, args.pre_cap, args.addr_cap, format(len(index), ",")))

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pending = args.out + ".partial"
    accepted = positives = 0
    t0 = time.time()

    with open(pending, "w", encoding="utf-8", buffering=1 << 22) as fo:
        fo.write((SCORE_HDR if args.score_only else HDR) + "\n")
        for src, path in (("S2", s2), ("S3", s3)):
            counts = array("I", bytes(4 * len(recs)))
            n_rows = 0
            with open(path, encoding="utf-8", errors="replace") as fh:
                next(fh, None)
                for line in fh:
                    n_rows += 1
                    if args.limit and n_rows > args.limit:
                        break
                    if n_rows % 200000 == 0:
                        log("%s: %s rows | %s pairs | %.0fs | %.0f rows/s"
                            % (src, format(n_rows, ","), format(accepted, ","),
                               time.time() - t0, n_rows / max(time.time() - t0, 1e-9)))
                    p = line.rstrip("\n").split("\t")
                    while len(p) < 4:
                        p.append("")
                    n2 = normalize_name(p[1])
                    if not n2:
                        continue
                    oid, c2 = p[0], cc(p[3])
                    atok2 = addr_tokens(p[2])
                    pins2 = set(pins(p[2]))
                    t2 = set(n2.split())
                    pre2 = {t[:PRE_MIN] for t in t2 if len(t) >= PRE_MIN}

                    # Phase 1 - visit posting lists ordered by (key strength,
                    # selectivity) and stop at --post-budget, so one common word
                    # cannot turn a single row into thousands of candidates.
                    lst_info = []
                    for w, k in keys_for(n2, c2, pins2, atok2):
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

                    # Phase 2 - the exact evidence score is evaluated only for
                    # the --pre-collect strongest candidates. Scoring every
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
                        # Address overlap: the only signal for transliteration
                        # pairs whose names share nothing. recs[i][5] holds the
                        # filtered address-token set (same normalisation as
                        # the candidate side).
                        a1 = recs[i][5]
                        if a1 and atok2:
                            sc += 1.5 * len(a1 & atok2)
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
                        lab = ""
                        if gt is not None:
                            lab = "1" if oid in gt.get(sid, ()) else "0"
                            if lab == "1":
                                positives += 1
                        if args.score_only:
                            fo.write("%s\t%s\t%g\t%s\n" % (sid, oid, sc, lab))
                        else:
                            r1 = {"n": recs[i][1],
                                  "atok": recs[i][5],
                                  "country": recs[i][3],
                                  "pins": recs[i][4]}
                            fv = features(r1, r2)
                            fo.write(sid + "\t" + oid + "\t" +
                                     "\t".join(("%.5g" % v) for v in fv) + "\t" + lab + "\n")
                        accepted += 1
                        if not args.score_only and accepted % 4000000 == 0:
                            fo.flush()

    el = time.time() - t0
    if os.path.exists(args.out):
        os.remove(args.out)
    os.replace(pending, args.out)
    log("DONE %s rows in %.1fs" % (format(accepted, ","), el))
    log("output: %s (%.2f GB)" % (args.out, os.path.getsize(args.out) / 1e9))
    if positives:
        log("positives: %s (%.4f%% of pairs)"
            % (format(positives, ","), 100.0 * positives / max(accepted, 1)))


def main():
    ap = argparse.ArgumentParser(description="one-pass union-key blocker")
    ap.add_argument("--split", default="test", choices=("train", "test"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--s1", default=None)
    ap.add_argument("--s2", default=None)
    ap.add_argument("--s3", default=None)
    ap.add_argument("--gt", default=None,
                    help="ground-truth tsv to label pairs (train split only)")
    ap.add_argument("--s1-limit", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="rows scanned per source file; 0 = all. WARNING: using a "
                         "limit for train but not test mis-calibrates the threshold")
    ap.add_argument("--tok-cap", type=int, default=600,
                    help="max Source-1 rows sharing a token key")
    ap.add_argument("--pre-cap", type=int, default=600,
                    help="max Source-1 rows sharing a token-prefix key")
    ap.add_argument("--addr-cap", type=int, default=400,
                        help="max Source-1 rows sharing an address-token key")
    ap.add_argument("--gate", type=float, default=3.0,
                    help="min combined evidence score to become a candidate")
    ap.add_argument("--strong", type=float, default=6.0,
                    help="score that is never cut by the budget")
    ap.add_argument("--budget", type=int, default=15,
                    help="max weak candidates per (source-1 row, source file)")
    ap.add_argument("--collect", type=int, default=40,
                    help="max candidates written per source-2/3 row")
    ap.add_argument("--pre-collect", type=int, default=40,
                    help="candidates given the exact evidence score")
    ap.add_argument("--post-budget", type=int, default=400,
                    help="max posting-list entries visited per source-2/3 row")
    ap.add_argument("--score-only", action="store_true",
                    help="write s1/other/evidence/label only, for recall probes")
    args = ap.parse_args()
    if args.out is None:
        args.out = "data/cache/%s_pairs.tsv" % args.split
    run(args)


if __name__ == "__main__":
    main()
