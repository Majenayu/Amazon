#!/usr/bin/env python3
"""Hash-join blocking core (import-only): 2 passes over S2/S3 per S1 slice."""
from __future__ import annotations
import os, sys, time
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))
ROOT = os.path.dirname(os.path.dirname(HERE))
from baseline.textnorm import (
    FEATURE_NAMES, features, make_record, normalize_name, pins, tokens)

HDR = "s1_id\tother_id\t" + "\t".join(FEATURE_NAMES) + "\tlabel"


def log(m):
    print("[hashjoin] %s" % m, file=sys.stderr, flush=True)


def norm_country(c):
    return (c or "").strip().lower()


def load_slice(s1_path, offset, limit):
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
            recs[p[0]] = make_record(p[0], p[1], p[2], p[3])
    return recs


def read_gt(path, keep):
    gt = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        next(fh, None)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if p[0] not in keep:
                continue
            gt[p[0]] = frozenset(x for x in p[1].split(",") if x) if len(p) > 1 and p[1] else frozenset()
    return gt
