#!/usr/bin/env python3
"""Candidate generation (blocking) + feature extraction, in ONE streaming pass.

Why one pass: the full dataset is 2.4 GB and this machine has ~5 GB of free RAM,
so we never hold Source-2/3 in memory. We only hold Source-1 (small enough) plus
a key index, and stream Source-2/Source-3 line by line, writing one feature row
per candidate pair.

Usage
-----
    # on the 11.8 MB sample (seconds)
    python src/baseline/block.py --split train --data-dir data/sample \\
        --out data/sample/cache/train_pairs.tsv
    python src/baseline/block.py --split test --data-dir data/sample \\
        --out data/sample/cache/test_pairs.tsv

    # full data, first 300k Source-1 rows (chunking keeps RAM flat)
    python src/baseline/block.py --split train --s1-limit 300000 --s1-offset 0 ...

Output: a TSV with header
    s1_id  other_id  name_exact name_jacc name_tri name_pre3 addr_jacc pin_eq
    country_eq len_ratio  label
label = 1/0 for --split train (ground truth), -1 for --split test.

The script prints blocking quality numbers (entity coverage + link recall),
which is the ceiling of everything downstream: if blocking misses a true match,
no model can recover it.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))

from baseline.textnorm import FEATURE_NAMES, features, make_record  # noqa: E402

HEADER = ["s1_id", "other_id"] + FEATURE_NAMES + ["label"]
EMPTY = frozenset()


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def read_gt(path: str, keep=None) -> dict:
    """Read the ground truth. If ``keep`` is given, only those entities are
    stored - on the full dataset the whole file is 2.2M rows / ~1 GB."""
    gt = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        next(fh, None)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if keep is not None and parts[0] not in keep:
                continue
            if len(parts) > 1 and parts[1]:
                gt[parts[0]] = frozenset(x for x in parts[1].split(",") if x)
            else:
                gt[parts[0]] = EMPTY
    return gt


def load_s1(path: str, offset: int, limit: int):
    recs = []
    s1_ids = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        next(fh, None)
        for i, line in enumerate(fh):
            if i < offset:
                continue
            if limit and len(recs) >= limit:
                break
            p = line.rstrip("\n").split("\t")
            while len(p) < 4:
                p.append("")
            recs.append(make_record(p[0], p[1], p[2], p[3]))
            s1_ids.append(p[0])
    return recs, s1_ids


def load_s1_ids(path: str, offset: int, limit: int):
    """Just the entity ids of the selected slice (cheap, no record building)."""
    ids = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        next(fh, None)
        for i, line in enumerate(fh):
            if i < offset:
                continue
            if limit and len(ids) >= limit:
                break
            ids.append(line.split("\t", 1)[0])
    return ids


def build_index(recs: list, max_key_span: int) -> dict:
    index = {}
    for i, rec in enumerate(recs):
        for k in rec["keys"]:
            index.setdefault(k, []).append(i)
    before = len(index)
    index = {k: v for k, v in index.items() if len(v) <= max_key_span}
    dropped = before - len(index)
    return index, dropped


def stream_pairs(path: str, index: dict, recs: list, gt, max_cand: int, out, stats: dict,
                 collect: int = 40, span_max: int = 100000):
    """Emit candidate pairs for one Source-2/3 file.

    Selection strategy - "rarest key first, quality floor":

      1. a record's keys are sorted by how many Source-1 rows they cover, and we
         walk them from the rarest.
      2. at most ``collect`` candidates are gathered per record, then they are
         ranked by shared name tokens and only the best ``max_cand`` (per
         Source-1 row, per source file) are written.
      3. QUALITY FLOOR: a pair is only written when it shares >=1 normalised
         name token OR shares a postal code. Pairs sharing only a 3-char
         prefix are pure noise: they filled per-row budgets with whoever
         arrived first in file order and starved true matches arriving later.
    """
    # counters are created per file, so each source (2 and 3) gets its OWN budget
    # for a Source-1 row - one source cannot eat the whole allowance.
    counters = array("I", bytes(4 * len(recs)))
    buf = []
    n_rows = 0
    fmt = "%.4f"
    t_start = time.time()

    def flush():
        if buf:
            out.write("\n".join(buf))
            out.write("\n")
            out.flush()
            buf.clear()

    with open(path, encoding="utf-8", errors="replace") as fh:
        next(fh, None)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            while len(p) < 4:
                p.append("")
            other_id = p[0]
            o = make_record(p[0], p[1], p[2], p[3])
            otok = o["ntok"]
            n_rows_scanned = stats.get("_scanned", 0) + 1
            stats["_scanned"] = n_rows_scanned
            if n_rows_scanned % 500000 == 0:
                el = time.time() - t_start
                log("    ... {:,} source rows scanned, {:,} pairs kept, {:.0f}s".format(
                    n_rows_scanned, stats["n_rows"], el))

            hits = []
            for k in o["keys"]:
                lst = index.get(k)
                if lst:
                    n = len(lst)
                    if n <= span_max:
                        hits.append((n, lst))
            if not hits:
                continue
            stats["considered"] += sum(n for n, _ in hits)
            hits.sort(key=lambda x: x[0])   # rarest key first

            cand = []
            seen = set()
            for n, lst in hits:
                for i in lst:
                    if i in seen:
                        continue
                    seen.add(i)
                    cand.append(i)
                    if len(cand) >= collect:
                        break
                if len(cand) >= collect:
                    break

            if len(cand) > max_cand > 0:
                cand.sort(key=lambda i: len(recs[i]["ntok"] & otok), reverse=True)
                stats["ranked"] += 1
                cand = cand[:max_cand]

            opins = set(o.get("pins") or ())
            for i in cand:
                if counters[i] >= max_cand:
                    continue
                r = recs[i]
                shared = len(r["ntok"] & otok)
                if shared == 0:
                    rpins = set(r.get("pins") or ())
                    if not (opins and rpins and (opins & rpins)):
                        continue  # prefix-only collision: noise, skip before budget
                is_pos = False
                if gt is not None:
                    is_pos = other_id in gt.get(recs[i]["id"], EMPTY)
                counters[i] += 1
                stats["cand"][i] = 1
                if is_pos:
                    stats["pos"][i] = 1
                    stats["n_pos"] += 1
                f = features(recs[i], o)
                row = [recs[i]["id"], other_id] + [fmt % x for x in f] + \
                      ["1" if is_pos else ("0" if gt is not None else "-1")]
                buf.append("\t".join(row))
                n_rows += 1
                stats["n_rows"] += 1
                if len(buf) >= 20_000:
                    flush()
    flush()
    return n_rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Blocking + feature extraction in one pass.")
    ap.add_argument("--split", choices=("train", "test"), required=True)
    ap.add_argument("--data-dir", default="data/sample", help="data folder (data/sample or data)")
    ap.add_argument("--out", required=True, help="output pairs TSV")
    ap.add_argument("--max-cand", type=int, default=150,
                    help="max negative candidates kept per Source-1 row (default 150)")
    ap.add_argument("--max-key-span", type=int, default=100000,
                    help="drop blocking keys covering more Source-1 rows than this (default 100000)")
    ap.add_argument("--collect", type=int, default=40,
                    help="candidates gathered per record before ranking (default 40)")
    ap.add_argument("--s1-limit", type=int, default=0, help="process only N Source-1 rows (0 = all)")
    ap.add_argument("--s1-offset", type=int, default=0, help="skip the first N Source-1 rows")
    args = ap.parse_args()

    split = args.split
    sub = os.path.join(args.data_dir, "train" if split == "train" else "test")
    s1_path = os.path.join(sub, "%s_source1.tsv" % split)
    src_paths = [os.path.join(sub, "%s_source%d.tsv" % (split, i)) for i in (2, 3)]
    gt_path = os.path.join(args.data_dir, "train", "train_ground_truth.tsv")

    for p in [s1_path] + src_paths:
        if not os.path.exists(p):
            log("missing file: %s" % p)
            return 1

    gt = None
    if split == "train":
        if not os.path.exists(gt_path):
            log("missing ground truth: %s" % gt_path)
            return 1
        log("loading ground truth (chunk only) ...")
        t0 = time.time()
        keep = set(load_s1_ids(s1_path, args.s1_offset, args.s1_limit))
        gt = read_gt(gt_path, keep=keep)
        del keep
        log("  {:,} labelled entities kept for this chunk in {:.1f}s".format(len(gt), time.time() - t0))

    log("loading Source 1 (offset=%d limit=%d) ..." % (args.s1_offset, args.s1_limit))
    t0 = time.time()
    recs, s1_ids = load_s1(s1_path, args.s1_offset, args.s1_limit)
    log("  %d Source-1 rows in %.1fs" % (len(recs), time.time() - t0))
    if not recs:
        log("no Source-1 rows selected")
        return 1

    t0 = time.time()
    index, dropped = build_index(recs, args.max_key_span)
    log("  key index: %d keys (%d dropped for being too common) in %.1fs"
        % (len(index), dropped, time.time() - t0))
    # the per-record key lists are now only needed inside the index - drop them
    for rec in recs:
        rec.pop("keys", None)

    stats = {
        "n_rows": 0,
        "n_pos": 0,
        "considered": 0,
        "ranked": 0,
        "cand": array("b", bytes(len(recs))),
        "pos": array("b", bytes(len(recs))),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()
    with open(args.out, "w", encoding="utf-8", newline="") as out:
        out.write("\t".join(HEADER) + "\n")
        for path in src_paths:
            if not os.path.exists(path):
                log("missing file: %s" % path)
                return 1
            log("streaming %s ..." % os.path.basename(path))
            sub_stats = {"n_rows": 0, "n_pos": 0, "considered": 0, "ranked": 0,
                         "cand": stats["cand"], "pos": stats["pos"]}
            stream_pairs(path, index, recs, gt, args.max_cand, out, sub_stats,
                         collect=args.collect, span_max=args.max_key_span)
            stats["n_rows"] += sub_stats["n_rows"]
            stats["n_pos"] += sub_stats["n_pos"]
            stats["considered"] += sub_stats["considered"]
            stats["ranked"] += sub_stats["ranked"]
            log("  running total: {:,} pairs ({:,} true matches, {:,} candidates considered)"
                .format(stats["n_rows"], stats["n_pos"], stats["considered"]))
    elapsed = time.time() - t0

    # ------------------------------------------------------------------ report
    n_s1 = len(recs)
    covered = sum(1 for x in stats["cand"] if x)
    log("")
    log("=== blocking report: %s ===" % split)
    log("  pairs written        : {:,}  in {:.1f}s".format(stats["n_rows"], elapsed))
    log("  Source-1 rows        : {:,}".format(n_s1))
    log("  candidates considered: {:,} ({:.1f} per Source-1 row before capping)"
        .format(stats["considered"], stats["considered"] / max(1, n_s1)))
    log("  with >=1 candidate   : {:,} ({:.2f}%)".format(covered, 100.0 * covered / n_s1))
    log("  output -> %s" % args.out)

    if gt is not None:
        with_matches = 0
        entities_hit = 0
        total_links = 0
        for i, sid in enumerate(s1_ids):
            links = gt.get(sid, EMPTY)
            if links:
                with_matches += 1
                total_links += len(links)
                if stats["pos"][i]:
                    entities_hit += 1
        log("  entities with matches : {:,}".format(with_matches))
        log("  >=1 TRUE match found  : {:,} ({:.2f}%)  <- entity recall"
            .format(entities_hit, 100.0 * entities_hit / max(1, with_matches)))
        log("  TRUE links recovered  : {:,}/{:,} ({:.2f}%)  <- link recall (BLOCKING CEILING)"
            .format(stats["n_pos"], total_links,
                    100.0 * stats["n_pos"] / max(1, total_links)))
        if stats["n_rows"]:
            log("  precision of pairs    : {:.2f}% are truly matching".format(
                100.0 * stats["n_pos"] / stats["n_rows"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

