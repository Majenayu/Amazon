"""v2 feature cache: parse-once S1 records + pre-split S2/S3 token rows.

Built alongside the running hj job - never touches hj_train.tsv.
S1 slice  : pickled dict {sid: (n, ntok, atok, country, pins)}
S2/S3 rows: one preparsed TSV each: oid, norm_name, tokens(;), pins(;), country
A v2 blocker can then skip normalize_name/pins/make_record entirely.
"""
from __future__ import annotations
import os, pickle, sys
HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))
ROOT = os.path.dirname(os.path.dirname(HERE))
from baseline.textnorm import make_record


def cache_s1(s1_path, out_path, offset=0, limit=0):
    recs = {}
    with open(s1_path, encoding="utf-8", errors="replace") as fh:
        next(fh, None)
        for i, line in enumerate(fh):
            if i < offset:
                continue
            if limit and len(recs) >= limit:
                break
            p = line.rstrip("\n").split("\t")
            while len(p) < 4:
                p.append("")
            r = make_record(p[0], p[1], p[2], p[3])
            recs[p[0]] = (r["n"], r["ntok"], r["atok"], r["country"], tuple(r["pins"]))
    with open(out_path, "wb") as f:
        pickle.dump(recs, f, protocol=4)
    print("cached %d S1 -> %s" % (len(recs), out_path), flush=True)
    return out_path


def preparse_other(src_path, out_path):
    """Parse one S2/S3 file once -> oid, n, toks(;), pins(;), country, atoks(;).

    STREAMING-SAFE: writes each line + flushes every 1M rows so a crash or
    OOM-kill never leaves a truncated file that looks complete. Also writes
    a .done marker only on success; v2block refuses to use a .pre without it.
    """
    import baseline.textnorm as T
    n = 0
    done = out_path + ".done"
    import os as _os
    if _os.path.exists(done):
        _os.remove(done)
    with open(src_path, encoding="utf-8", errors="replace") as fin, \
            open(out_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i == 0:
                continue
            p = line.rstrip("\n").split("\t")
            while len(p) < 4:
                p.append("")
            nn = T.normalize_name(p[1])
            fout.write(p[0] + "\t" + nn + "\t" + ";".join(nn.split())
                       + "\t" + ";".join(T.pins(p[2])) + "\t"
                       + (p[3] or "").strip().lower() + "\t"
                       + ";".join((p[2] or "").lower().split()[:40]) + "\n")
            n += 1
            if n % 1000000 == 0:
                fout.flush()
                print("  preparsed %dM" % (n // 1000000), flush=True)
    print("preparsed %d -> %s" % (n, out_path), flush=True)
    open(done, "w").write(str(n) + "\n")
    return out_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1")
    ap.add_argument("--s1-out")
    ap.add_argument("--src")
    ap.add_argument("--src-out")
    a = ap.parse_args()
    if a.s1:
        cache_s1(a.s1, a.s1_out)
    if a.src:
        preparse_other(a.src, a.src_out)
