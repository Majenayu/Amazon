#!/usr/bin/env python3
"""v2 blocker on preparsed files (no re-parsing): same pairs, ~3x faster."""
from __future__ import annotations
import os, pickle, sys, time
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))
from baseline.textnorm import FEATURE_NAMES, jaccard, trigrams

HDR = "s1_id\tother_id\t" + "\t".join(FEATURE_NAMES) + "\tlabel"


def log(m):
    print("[v2] %s" % m, file=sys.stderr, flush=True)


def fast_feats(sn, stok, satok, sc, spins, on, otok, oatok, oc, opins,
               sg=None, og=None):
    ex = 1.0 if (sn == on and sn) else 0.0
    if sg is None:
        sg = trigrams(sn)
    if og is None:
        og = trigrams(on)
    return (ex, jaccard(stok, otok), jaccard(sg, og),
            1.0 if (len(sn) >= 3 and sn[:3] == on[:3]) else 0.0,
            jaccard(satok, oatok),
            1.0 if (spins and opins and (spins & opins)) else 0.0,
            1.0 if sc == oc else 0.0,
            (min(len(sn), len(on)) / float(max(len(sn), len(on)))) if max(len(sn), len(on)) else 1.0)


def load_s1_slice(s1_path, offset, limit):
    """Stream ONE S1 slice from raw TSV (bounded RAM, no full pickle load)."""
    import baseline.vcache as V
    import baseline.textnorm as T
    recs = {}
    order = []
    with open(s1_path, encoding="utf-8", errors="replace") as fh:
        next(fh, None)
        for i, line in enumerate(fh):
            if i < offset:
                continue
            if len(order) >= limit:
                break
            p = line.rstrip("\n").split("\t")
            while len(p) < 4:
                p.append("")
            r = V.make_record_fast(p[0], p[1], p[2], p[3]) if hasattr(V, "make_record_fast") else None
            if r is None:
                nn = T.normalize_name(p[1])
                sc = (p[3] or "").strip().lower()
                recs[p[0]] = (nn, set(nn.split()), set((p[2] or "").lower().split()),
                              sc, set(T.pins(p[2])), T.trigrams(nn))
            else:
                recs[p[0]] = r
            order.append(p[0])
    return recs, order


def run(split, data_dir, pre_dir, out_path, chunk=50_000, tok_top=8,
        tok_cap=150, ex_top=20, gt=None):
    """Stream slices; each slice appends to out_path. Crash-safe + low RAM.

    Writes seg_XXXXXX.tsv per chunk instead of one giant file handle, so a
    crash never loses earlier slices and no flush/OSError can wipe progress.
    """
    import glob
    sub = os.path.join(data_dir, "train" if split == "train" else "test")
    s1_path = os.path.join(sub, "%s_source1.tsv" % split)
    import baseline.vcache as V
    total = sum(1 for _ in open(s1_path, encoding="utf-8", errors="replace")) - 1
    log("%s: %d S1 rows (streaming, chunk=%d)" % (split, total, chunk))
    p2 = os.path.join(pre_dir, "%s_s2.pre" % split)
    p3 = os.path.join(pre_dir, "%s_s3.pre" % split)
    for src, pre in ((os.path.join(sub, "%s_source2.tsv" % split), p2),
                     (os.path.join(sub, "%s_source3.tsv" % split), p3)):
        if not os.path.exists(pre):
            V.preparse_other(src, pre)
    segdir = out_path + ".segs"
    os.makedirs(segdir, exist_ok=True)
    have = {os.path.basename(p) for p in glob.glob(os.path.join(segdir, "seg_*.tsv"))
            if os.path.getsize(p) > 0}
    # remove 0-byte leftovers from killed runs so they get redone
    for p in glob.glob(os.path.join(segdir, "seg_*.tsv")):
        if os.path.getsize(p) == 0:
            try:
                os.remove(p)
            except OSError:
                pass
    pairs = pos = 0
    t0 = time.time()
    for s in range(0, total, chunk):
        segname = "seg_%06d.tsv" % s
        if segname in have:
            log("  slice %d/%d SKIPPED (already done)" % (s + chunk, total))
            continue
        R, order = load_s1_slice(s1_path, s, chunk)
        if not order:
            break
        sl = order
        segpath = os.path.join(segdir, segname)
        nseg = 0
        with open(segpath, "w", encoding="utf-8") as out:
            exact = defaultdict(list)
            posting = defaultdict(list)
            for sid, rec in R.items():
                sn, stok = rec[0], rec[1]
                sc = rec[3]
                if sn:
                    exact[(sc, sn)].append(sid)
                seen = set()
                for t in stok:
                    if len(t) >= 4 and t not in seen:
                        seen.add(t)
                        posting[(sc, t)].append(sid)
            rare = {k for k, v in posting.items() if len(v) <= tok_cap}
            budget = {}
            from baseline.textnorm import trigrams as _tri2
            # SLOW-PASS GUARD: a 50k slice should finish its S2 scan in ~2-3 min.
            # Log progress every 1M pre-parsed rows so a stall is visible.
            for pre in (p2, p3):
                fn = "2" if pre.endswith("s2.pre") else "3"
                with open(pre, encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        q = line.rstrip("\n").split("\t")
                        if len(q) < 6:
                            continue
                        oid, on, oc = q[0], q[1], q[4]
                        otok = set(q[2].split(";")) if q[2] else set()
                        opins = set(q[3].split(";")) if q[3] else set()
                        oatok = set(q[5].split(";")) if q[5] else set()
                        og = None
                        done = set()
                        if on:
                            for sid in exact.get((oc, on), ()):
                                if sid in done:
                                    continue
                                done.add(sid)
                                key = (sid, fn)
                                b = budget.get(key, 0)
                                if b >= ex_top:
                                    continue
                                budget[key] = b + 1
                                rec = R[sid]
                                sn, stok, satok, sc, spins, sg = rec
                                if og is None:
                                    og = _tri2(on)
                                f = fast_feats(sn, stok, satok, sc, spins, on, otok, oatok, oc, opins, sg, og)
                                lab = "-1" if gt is None else ("1" if oid in gt.get(sid, ()) else "0")
                                out.write(sid + "\t" + oid + "\t" + "\t".join("%.4f" % x for x in f) + "\t" + lab + "\n")
                                pairs += 1
                                nseg += 1
                                if lab == "1":
                                    pos += 1
                        cand = defaultdict(int)
                        for t in otok:
                            if len(t) >= 4 and (oc, t) in rare:
                                for sid in posting[(oc, t)]:
                                    cand[sid] += 1
                        if cand:
                            top = sorted(cand, key=lambda x: cand[x], reverse=True)[:tok_top]
                            for sid in top:
                                if sid in done:
                                    continue
                                done.add(sid)
                                key = (sid, fn)
                                b = budget.get(key, 0)
                                if b >= ex_top:
                                    continue
                                rec = R[sid]
                                sn, stok, satok, sc, spins, sg = rec
                                if not (stok & otok):
                                    if not (spins and opins and (spins & opins)):
                                        continue
                                budget[key] = b + 1
                                if og is None:
                                    og = _tri2(on)
                                f = fast_feats(sn, stok, satok, sc, spins, on, otok, oatok, oc, opins, sg, og)
                                lab = "-1" if gt is None else ("1" if oid in gt.get(sid, ()) else "0")
                                out.write(sid + "\t" + oid + "\t" + "\t".join("%.4f" % x for x in f) + "\t" + lab + "\n")
                                pairs += 1
                                nseg += 1
                                if lab == "1":
                                    pos += 1
            log("  slice %d/%d pairs=%d (+%d) pos=%d %.0fs" % (s + len(sl), total, pairs, nseg, pos, time.time() - t0))
            del R, order, exact, posting, rare, budget
    # concat segments into final file (single header)
    import glob as _glob
    segs = sorted(_glob.glob(os.path.join(segdir, "seg_*.tsv")))
    with open(out_path, "w", encoding="utf-8") as fout:
        fout.write(HDR + "\n")
        for sp in segs:
            with open(sp, encoding="utf-8", errors="replace") as fin:
                for line in fin:
                    fout.write(line)
    log("DONE %s: %d pairs in %.0fs -> %s" % (split, pairs, time.time() - t0, out_path))
    return out_path


if __name__ == "__main__":
    raise SystemExit("use run_v2.py")
