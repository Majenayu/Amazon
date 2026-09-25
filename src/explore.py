#!/usr/bin/env python3
"""Streaming EDA over the FULL dataset (~2.5 GB / 26M rows) with low memory.

Rules this script follows (and any script in this repo should follow):

* never loads a whole source file into memory - reads in 200k-row chunks;
* never prints raw dataset rows;
* prints only small summary numbers you can paste straight back into an AI prompt.

What you get:
  * row counts per file
  * country distribution per file (checks for France in the test set)
  * missing / empty rates for name and address
  * name & address length percentiles and non-ASCII rate (first 400k rows)
  * ground-truth match-count distribution and the singleton rate, which sets the
    "predict nothing" baseline of the macro F_0.5 metric

Usage:
    python src/explore.py
    python src/explore.py --save notes/eda_summary.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

CHUNK = 200_000
EXPENSIVE_ROW_BUDGET = 400_000  # length / non-ASCII stats on the first N rows only
NON_ASCII = r"[^\x00-\x7F]"


def pct(n: int, d: int) -> str:
    return "%.2f%%" % (100.0 * n / d) if d else "n/a"


def summarize(lens) -> str:
    if not len(lens):
        return "n/a"
    s = pd.Series(lens)
    return "p10=%.0f  median=%.0f  p90=%.0f  max=%.0f" % (
        s.quantile(0.1), s.median(), s.quantile(0.9), s.max())


def profile_source(path: str, label: str, out: list) -> None:
    rows = 0
    empty_name = empty_addr = 0
    countries = Counter()
    name_lens: list = []
    addr_lens: list = []
    non_ascii = 0
    profiled = 0

    reader = pd.read_csv(path, sep="\t", chunksize=CHUNK, dtype=str,
                         keep_default_na=False, na_filter=False, encoding="utf-8")
    for chunk in reader:
        rows += len(chunk)
        countries.update(chunk["country"].value_counts().to_dict())

        name = chunk["business_name"]
        addr = chunk["business_address"]
        empty_name += int((name.str.strip() == "").sum())
        empty_addr += int((addr.str.strip() == "").sum())

        if profiled < EXPENSIVE_ROW_BUDGET:
            take = chunk.head(EXPENSIVE_ROW_BUDGET - profiled)
            profiled += len(take)
            name_lens.extend(take["business_name"].str.len().tolist())
            addr_lens.extend(take["business_address"].str.len().tolist())
            non_ascii += int(take["business_name"].str.contains(NON_ASCII, regex=True).sum())

    top = ", ".join("%s=%s (%s)" % (k or "<empty>", "{:,}".format(v), pct(v, rows))
                    for k, v in countries.most_common(8))
    out.append("--- %s: %s" % (label, os.path.basename(path)))
    out.append("    rows            : {:,}".format(rows))
    out.append("    empty name      : %s" % pct(empty_name, rows))
    out.append("    empty address   : %s" % pct(empty_addr, rows))
    out.append("    countries       : %s" % top)
    out.append("    name length     : %s" % summarize(name_lens))
    out.append("    address length  : %s" % summarize(addr_lens))
    out.append("    non-ASCII names : %s (first %s rows)"
               % (pct(non_ascii, profiled), "{:,}".format(profiled)))
    out.append("")


def profile_ground_truth(path: str, out: list) -> None:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                     na_filter=False, encoding="utf-8")
    total = len(df)
    ids = df["matched_entity_ids"].str.strip()
    has_match = ids != ""
    matched_rows = int(has_match.sum())
    counts = ids[has_match].str.count(",").astype(int) + 1

    dist = Counter(int(c) for c in counts.tolist())
    singletons = total - matched_rows
    total_matches = int(counts.sum())

    eight_plus = sum(v for k, v in dist.items() if k >= 8)
    shown = ", ".join("%d: %s" % (k, "{:,}".format(v))
                      for k, v in sorted(dist.items()) if k < 8)
    if eight_plus:
        shown += ", 8+: {:,}".format(eight_plus)

    out.append("--- ground truth: train_ground_truth.tsv")
    out.append("    S1 entities           : {:,}".format(total))
    out.append("    with >=1 match        : {:,} ({})".format(matched_rows, pct(matched_rows, total)))
    out.append("    singletons (no match) : {:,} ({})".format(singletons, pct(singletons, total)))
    out.append("    total true matches    : {:,}  (avg {:.2f} per matched entity)".format(
        total_matches, total_matches / max(1, matched_rows)))
    out.append("    matches per entity    : %s" % shown)
    out.append("    predict-all-empty baseline (macro F_0.5): %s" % pct(singletons, total))
    out.append("")


def main() -> int:
    ap = argparse.ArgumentParser(description="Streaming EDA over the full dataset.")
    ap.add_argument("--data-dir", default=DATA_DIR, help="path to the data/ folder")
    ap.add_argument("--save", default=None, help="also write the report to this file")
    args = ap.parse_args()

    train = os.path.join(args.data_dir, "train")
    test = os.path.join(args.data_dir, "test")
    plan = [
        (os.path.join(train, "train_source1.tsv"), "train S1"),
        (os.path.join(train, "train_source2.tsv"), "train S2"),
        (os.path.join(train, "train_source3.tsv"), "train S3"),
        (os.path.join(test, "test_source1.tsv"), "test S1"),
        (os.path.join(test, "test_source2.tsv"), "test S2"),
        (os.path.join(test, "test_source3.tsv"), "test S3"),
    ]
    missing = [p for p, _ in plan if not os.path.exists(p)]
    if missing:
        print("Missing files:", file=sys.stderr)
        for p in missing:
            print("  - " + p, file=sys.stderr)
        return 1

    out: list = []
    for path, label in plan:
        print("scanning %s: %s ..." % (label, os.path.basename(path)), file=sys.stderr)
        profile_source(path, label, out)

    gt = os.path.join(train, "train_ground_truth.tsv")
    if os.path.exists(gt):
        print("scanning ground truth ...", file=sys.stderr)
        profile_ground_truth(gt, out)

    report = "\n".join(out)
    print(report)
    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")
        print("saved -> %s" % args.save, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

