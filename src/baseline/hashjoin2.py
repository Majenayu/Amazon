from collections import defaultdict
import os
from baseline.hashjoin import norm_country
from baseline.textnorm import features, make_record, normalize_name, pins, tokens


def emit(out, r, o, label):
    f = features(r, o)
    out.write(r["id"] + "\t" + o["id"] + "\t" + "\t".join("%.4f" % x for x in f) + "\t" + label + "\n")


def block_slice(split, sub, recs, gt, out, stats, tok_cap=150, tok_top=8, ex_top=20):
    """Two hash passes over S2+S3 for one S1 slice."""
    exact = defaultdict(list)
    posting = defaultdict(list)
    for sid, r in recs.items():
        c = norm_country(r["country"])
        if r["n"]:
            exact[(c, r["n"])].append(sid)
        seen = set()
        for t in r["ntok"]:
            if len(t) >= 4 and t not in seen:
                seen.add(t)
                posting[(c, t)].append(sid)
    rare = {k for k, v in posting.items() if len(v) <= tok_cap}
    stats["rare"] = len(rare)
    budget = {}
    for fn in ("2", "3"):
        path = os.path.join(sub, "%s_source%s.tsv" % (split, fn))
        with open(path, encoding="utf-8", errors="replace") as fh:
            next(fh, None)
            for line in fh:
                p = line.rstrip("\n").split("\t")
                while len(p) < 4:
                    p.append("")
                oid, nm, ad, co = p[0], p[1], p[2], p[3]
                c = norm_country(co)
                n = normalize_name(nm)
                done = set()
                if n:
                    for sid in exact.get((c, n), ()):
                        if sid in done:
                            continue
                        done.add(sid)
                        key = (sid, fn)
                        b = budget.get(key, 0)
                        if b >= ex_top:
                            continue
                        budget[key] = b + 1
                        r = recs[sid]
                        o = make_record(oid, nm, ad, co)
                        lab = "-1" if gt is None else ("1" if oid in gt.get(sid, ()) else "0")
                        emit(out, r, o, lab)
                        stats["pairs"] += 1
                        if lab == "1":
                            stats["pos"] += 1
                cand = defaultdict(int)
                for t in tokens(n):
                    if len(t) >= 4 and (c, t) in rare:
                        for sid in posting[(c, t)]:
                            cand[sid] += 1
                if cand:
                    ranked = sorted(cand, key=lambda s: cand[s], reverse=True)[:tok_top]
                    op = set(pins(ad))
                    for sid in ranked:
                        if sid in done:
                            continue
                        done.add(sid)
                        key = (sid, fn)
                        b = budget.get(key, 0)
                        if b >= ex_top:
                            continue
                        r = recs[sid]
                        o = make_record(oid, nm, ad, co)
                        if len(r["ntok"] & o["ntok"]) == 0:
                            rp = set(r.get("pins") or ())
                            if not (op and rp and (op & rp)):
                                continue
                        budget[key] = b + 1
                        lab = "-1" if gt is None else ("1" if oid in gt.get(sid, ()) else "0")
                        emit(out, r, o, lab)
                        stats["pairs"] += 1
                        if lab == "1":
                            stats["pos"] += 1
