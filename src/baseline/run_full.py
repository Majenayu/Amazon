#!/usr/bin/env python3
"""Run the whole baseline on the FULL 2.5 GB dataset, chunk by chunk.

Design constraints on this machine: ~1.7 GB of free commit memory, so

* Source-1 is processed in chunks of --chunk rows (fresh process per chunk, so
  memory is released between chunks);
* everything runs sequentially (no parallel workers);
* every finished chunk is skipped on restart, so an interrupted run resumes.

Phases:

    python src/baseline/run_full.py --phase block-train
    python src/baseline/run_full.py --phase train
    python src/baseline/run_full.py --phase block-test
    python src/baseline/run_full.py --phase predict
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(ROOT, "src", "baseline")
CACHE = os.path.join(ROOT, "data", "cache")
LOGS = os.path.join(ROOT, "logs")
MODEL = os.path.join(ROOT, "models", "full")


def log(m):
    print(m, file=sys.stderr, flush=True)


def run(cmd, log_path, tag):
    log("[%s] start" % tag)
    t0 = time.time()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
    log("[%s] exit=%d in %.0fs -> %s" % (tag, rc, time.time() - t0, log_path))
    return rc


def s1_count(path):
    n = 0
    with open(path, "rb") as fh:
        next(fh, None)
        for _ in fh:
            n += 1
    return n


def concat(files, out_path):
    if os.path.exists(out_path):
        os.remove(out_path)
    first = True
    with open(out_path, "w", encoding="utf-8", newline="") as out:
        for f in files:
            with open(f, encoding="utf-8", errors="replace") as inp:
                for i, line in enumerate(inp):
                    if i == 0 and not first:
                        continue
                    out.write(line)
                    if i == 0:
                        first = False
    log("concatenated %d files -> %s (%.1f GB)" % (
        len(files), out_path, os.path.getsize(out_path) / 1e9))


def block(split, data_dir, chunk, max_cand, collect, n_s1, tag):
    offsets = list(range(0, n_s1, chunk))
    outs = []
    for i, off in enumerate(offsets):
        out = os.path.join(CACHE, "%s_chunk_%02d.tsv" % (split, i))
        outs.append(out)
        if os.path.exists(out) and os.path.getsize(out) > 1024:
            log("skip %s chunk %d (already done)" % (split, i))
            continue
        cmd = [sys.executable, os.path.join(BASE, "block.py"),
               "--split", split, "--data-dir", data_dir, "--out", out,
               "--s1-offset", str(off), "--s1-limit", str(chunk),
               "--max-cand", str(max_cand), "--collect", str(collect)]
        rc = run(cmd, os.path.join(LOGS, "%s_chunk_%02d.log" % (tag, i)),
                 "%s-%d" % (split, i))
        if rc != 0:
            log("FAILED %s chunk %d - stopping" % (split, i))
            return rc, outs
    return 0, outs


def main():
    ap = argparse.ArgumentParser(description="Full-dataset baseline runner.")
    ap.add_argument("--phase", required=True,
                    choices=("block-train", "train", "block-test", "predict", "all"))
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--chunk", type=int, default=400000)
    ap.add_argument("--max-cand", type=int, default=10)
    ap.add_argument("--collect", type=int, default=40)
    args = ap.parse_args()

    data_dir = args.data_dir if os.path.isabs(args.data_dir) \
        else os.path.join(ROOT, args.data_dir)
    os.makedirs(CACHE, exist_ok=True)
    py = sys.executable

    n_train = s1_count(os.path.join(data_dir, "train", "train_source1.tsv"))
    n_test = s1_count(os.path.join(data_dir, "test", "test_source1.tsv"))
    log("train S1 {:,} | test S1 {:,} | chunk {:,}".format(n_train, n_test, args.chunk))

    phases = ["block-train", "train", "block-test", "predict"] \
        if args.phase == "all" else [args.phase]

    for ph in phases:
        if ph == "block-train":
            rc, outs = block("train", data_dir, args.chunk, args.max_cand,
                             args.collect, n_train, "train")
            if rc:
                return rc
            concat(outs, os.path.join(CACHE, "train_pairs.tsv"))
        elif ph == "train":
            rc = run([py, os.path.join(BASE, "train.py"),
                      "--pairs", os.path.join(CACHE, "train_pairs.tsv"),
                      "--s1", os.path.join(data_dir, "train", "train_source1.tsv"),
                      "--gt", os.path.join(data_dir, "train", "train_ground_truth.tsv"),
                      "--model-out", MODEL],
                     os.path.join(LOGS, "train.log"), "train")
            if rc:
                return rc
        elif ph == "block-test":
            rc, outs = block("test", data_dir, args.chunk, args.max_cand,
                             args.collect, n_test, "test")
            if rc:
                return rc
            concat(outs, os.path.join(CACHE, "test_pairs.tsv"))
        elif ph == "predict":
            rc = run([py, os.path.join(BASE, "predict.py"),
                      "--pairs", os.path.join(CACHE, "test_pairs.tsv"),
                      "--model", MODEL,
                      "--s1", os.path.join(data_dir, "test", "test_source1.tsv"),
                      "--out-dir", os.path.join(ROOT, "output")],
                     os.path.join(LOGS, "predict.log"), "predict")
            if rc:
                return rc
            rc = run([py, os.path.join(ROOT, "src", "validate_submission.py"),
                      "--matching", os.path.join(ROOT, "output", "matching_results.tsv"),
                      "--candidate", os.path.join(ROOT, "output", "candidate_pairs.tsv"),
                      "--test-dir", os.path.join(data_dir, "test")],
                     os.path.join(LOGS, "validate.log"), "validate")
            if rc:
                return rc
    log("phase %s complete" % args.phase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

