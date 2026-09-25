#!/usr/bin/env python3
"""Train the matcher and pick the probability threshold maximising macro F_0.5.

Scalable version: pairs are read in chunks so the full-data run (tens of
millions of pairs) never needs to fit in RAM at once.

* training rows  = every TRUE match + a random 15% sample of the noise
  (the model only needs to learn the ranking; keeping all noise wastes RAM)
* validation rows = ALL pairs of the held-out entities, at full density - the
  threshold must be tuned on exactly the density we will serve at, otherwise
  the chosen threshold is wrong.

Metric, per Source-1 entity, then averaged:

    m = |true|, k = |predicted|, t = |correct|
    m == 0 and k == 0 -> 1.0        (correct singleton)
    m == 0 and k >  0 -> 0.0        (false merge - very expensive)
    otherwise         -> 1.25*t / (0.25*m + k)

Entities with no candidate at all are still scored, exactly like the leaderboard.

Usage:
    python src/baseline/train.py --pairs data/cache/train_pairs.tsv \\
        --s1 data/train/train_source1.tsv --gt data/train/train_ground_truth.tsv \\
        --model-out models/full
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))

from baseline.textnorm import FEATURE_NAMES  # noqa: E402

EMPTY = frozenset()
NEG_KEEP = 0.15  # fraction of noise rows kept for training


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def split_of(s1_id: str) -> str:
    """Stable hash split: ~20% of entities go to validation."""
    return "val" if (zlib.crc32(s1_id.encode("utf-8")) % 5 == 0) else "trn"


def read_gt(path: str) -> dict:
    gt = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        next(fh, None)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) > 1 and p[1]:
                gt[p[0]] = frozenset(x for x in p[1].split(",") if x)
            else:
                gt[p[0]] = EMPTY
    return gt


def read_s1_ids(path: str) -> list:
    ids = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        next(fh, None)
        for line in fh:
            ids.append(line.split("\t", 1)[0])
    return ids


def sweep(val_p: np.ndarray, val_y: np.ndarray, val_code: np.ndarray,
          gt_len: np.ndarray) -> tuple:
    """Vectorised threshold sweep. Returns (best_threshold, best_score, curve)."""
    n = len(gt_len)
    if len(val_p) == 0 or n == 0:
        return 1.0, float((gt_len == 0).mean()) if n else 0.0, []
    qs = np.unique(np.round(np.quantile(val_p, np.linspace(0.0, 1.0, 101)), 4))
    best_t, best_s, curve = 1.0, -1.0, []
    for t in qs:
        mask = val_p >= t
        k = np.bincount(val_code[mask], minlength=n).astype(np.float64)
        correct = np.bincount(val_code[mask], weights=val_y[mask], minlength=n)
        with np.errstate(divide="ignore", invalid="ignore"):
            scored = np.where(gt_len == 0, (k == 0).astype(np.float64),
                              np.where(k > 0, 1.25 * correct / (0.25 * gt_len + k), 0.0))
        s = float(scored.mean())
        curve.append((float(t), s))
        if s > best_s:
            best_s, best_t = s, float(t)
    return best_t, best_s, curve


def main() -> int:
    ap = argparse.ArgumentParser(description="Train matcher + tune F_0.5 threshold (streaming).")
    ap.add_argument("--pairs", required=True, help="pairs TSV from block.py --split train")
    ap.add_argument("--s1", required=True, help="train_source1.tsv")
    ap.add_argument("--gt", required=True, help="train_ground_truth.tsv")
    ap.add_argument("--model-out", required=True, help="output prefix, e.g. models/full")
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--neg-keep", type=float, default=NEG_KEEP)
    args = ap.parse_args()

    for p in (args.pairs, args.s1, args.gt):
        if not os.path.exists(p):
            log("missing file: %s" % p)
            return 1

    # ---- validation entities (hash split, reproducible, no seed needed) -------
    t0 = time.time()
    val_ids, code_map, inv = [], {}, []
    with open(args.s1, encoding="utf-8", errors="replace") as fh:
        next(fh, None)
        for line in fh:
            sid = line.split("\t", 1)[0]
            if split_of(sid) == "val":
                code_map[sid] = len(inv)
                inv.append(sid)
                val_ids.append(sid)
    log("validation entities: {:,} ({:.1f}% of Source-1)".format(len(val_ids), 100.0 * len(val_ids) / max(1, len(inv))))

    # ---- keep ONLY the ground truth of validation entities (saves ~700 MB) ----
    val_gt = {}
    with open(args.gt, encoding="utf-8", errors="replace") as fh:
        next(fh, None)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if p[0] in code_map:
                val_gt[p[0]] = frozenset(x for x in p[1].split(",")) if len(p) > 1 and p[1] else EMPTY
    gt_len = np.array([len(val_gt.get(sid, EMPTY)) for sid in val_ids], dtype=np.int32)
    n_singleton = int((gt_len == 0).sum())
    log("  singleton rate {:.2f}% = predict-all-empty floor".format(
        100.0 * n_singleton / max(1, len(val_ids))))

    # ---- stream the pairs file -------------------------------------------------
    feats = FEATURE_NAMES
    split_cache = {}
    rng = np.random.default_rng(0)
    fit_X, fit_y, val_X, val_y, val_code = [], [], [], [], []
    n_pos = n_neg = n_val_rows = 0
    t1 = time.time()
    reader = pd.read_csv(args.pairs, sep="\t", chunksize=args.chunksize,
                         dtype={"s1_id": str, "other_id": str})
    for ci, chunk in enumerate(reader):
        if "label" in chunk.columns:
            chunk = chunk[chunk["label"] >= 0]
        if chunk.empty:
            continue

        def get_split(sid):
            v = split_cache.get(sid)
            if v is None:
                v = split_of(sid)
                split_cache[sid] = v
            return v

        sp = chunk["s1_id"].map(get_split)

        v = chunk[sp == "val"]
        if len(v):
            val_code.append(v["s1_id"].map(code_map).to_numpy(dtype=np.int32))
            val_X.append(v[feats].to_numpy(dtype=np.float32))
            val_y.append(v["label"].to_numpy(dtype=np.int8))
            n_val_rows += len(v)

        tr = chunk[sp == "trn"]
        if len(tr):
            pos = tr[tr["label"] == 1]
            neg = tr[tr["label"] == 0]
            n_pos += len(pos)
            if len(neg) and args.neg_keep < 1.0:
                neg = neg[rng.random(len(neg)) < args.neg_keep]
            n_neg += len(neg)
            keep = pd.concat([pos, neg], ignore_index=True) if len(neg) else pos
            fit_X.append(keep[feats].to_numpy(dtype=np.float32))
            fit_y.append(keep["label"].to_numpy(dtype=np.int8))
        if (ci + 1) % 5 == 0:
            log("  chunk {:,} | fit rows {:,} | val rows {:,}".format(
                ci + 1, n_pos + n_neg, n_val_rows))
    log("pairs streamed in {:.1f}s".format(time.time() - t1))

    if not fit_X:
        log("no training pairs found")
        return 1

    X = np.vstack(fit_X)
    y = np.concatenate(fit_y)
    del fit_X, fit_y
    log("training set: {:,} rows ({:,} positive, {:,} sampled noise) in {:.1f}s"
        .format(len(y), int((y == 1).sum()), int((y == 0).sum()), time.time() - t0))

    from sklearn.ensemble import HistGradientBoostingClassifier
    log("training HistGradientBoosting ({} features) ...".format(len(feats)))
    clf = HistGradientBoostingClassifier(
        max_iter=args.max_iter, learning_rate=0.1, max_leaf_nodes=63,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        random_state=0)
    clf.fit(X, y)
    del X, y

    best_t, best_s, curve = 0.5, 0.0, []
    if val_code:
        VC = np.concatenate(val_code)
        VX = np.vstack(val_X)
        VY = np.concatenate(val_y)
        del val_X, val_y
        val_p = clf.predict_proba(VX)[:, 1].astype(np.float32)
        best_t, best_s, curve = sweep(val_p, VY, VC, gt_len)
        mask = val_p >= best_t
        log("")
        log("=== threshold sweep on {:,} validation entities ({:,} pairs) ==="
            .format(len(gt_len), len(val_p)))
        log("  predict-all-empty score : {:.4f}".format(n_singleton / max(1, len(gt_len))))
        log("  BEST threshold          : {:.4f}".format(best_t))
        log("  BEST macro F_0.5        : {:.4f}".format(best_s))
        k = np.bincount(VC[mask], minlength=len(gt_len))
        log("  entities predicted non-empty at best t: {:,} ({:.1f}%)"
            .format(int((k > 0).sum()), 100.0 * (k > 0).mean()))
        for t, s in curve[::10]:
            log("    t={:.4f}  F_0.5={:.4f}".format(t, s))
    else:
        log("no validation pairs found - threshold left at 0.5")

    import joblib
    prefix = args.model_out
    os.makedirs(os.path.dirname(os.path.abspath(prefix)) or ".", exist_ok=True)
    joblib.dump({"model": clf, "features": feats, "threshold": best_t}, prefix + ".joblib")
    meta = {
        "pairs": args.pairs,
        "n_fit_pos": int(n_pos),
        "n_fit_neg_sampled": int(n_neg),
        "n_val_entities": int(len(gt_len)),
        "n_val_pairs": int(len(val_code)) if val_code else 0,
        "singleton_rate": float(n_singleton / max(1, len(gt_len))),
        "best_threshold": float(best_t),
        "best_macro_f05": float(best_s),
        "neg_keep": float(args.neg_keep),
        "features": feats,
        "curve": [[float(t), float(s)] for t, s in curve],
    }
    with open(prefix + "_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    log("saved -> %s.joblib , %s_meta.json" % (prefix, prefix))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


