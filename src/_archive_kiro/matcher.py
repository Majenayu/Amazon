#!/usr/bin/env python3
"""Stage 2b: LightGBM matcher. Train on labeled feature shards; score candidates.

Design choices tied to the red-team review:
  * ONE model with an is_s3 source-indicator feature, not two specialists (D9):
    splitting halves data per model with no evidence the decision boundary differs.
  * Negatives come from retrieval output (the hard confusables), which is exactly
    what the matcher must separate (D20). The label is 1 iff the candidate is a true
    match for that S1, so every non-matching retrieved candidate is a hard negative.
  * country is NOT a filter here and NOT one-hot to {US,India}; it never enters the
    model as a raw label, so an unseen label (France) cannot break inference (S3.3).

Memory safety: feature shards are read one at a time and concatenated only as float32
numpy; on 64 GB this is fine for the full candidate volume, but if it ever grows too
large the same code path supports training on a subsample of shards.

Outputs: a saved LightGBM model (model.txt) and, at scoring time, parquet shards with
(s1_id, tgt_id, score).
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FEATURE_NAMES  # noqa: E402


def _load_feature_shards(feat_dir, with_label):
    """Load all feature shards into X (float32), and y/ids as needed."""
    files = sorted(glob.glob(os.path.join(feat_dir, "*.parquet")))
    if not files:
        raise SystemExit("no feature shards in %s" % feat_dir)
    xs, ys, s1s, tgts = [], [], [], []
    for f in files:
        df = pd.read_parquet(f)
        xs.append(df[FEATURE_NAMES].to_numpy(dtype=np.float32))
        s1s.append(df["s1_id"].to_numpy())
        tgts.append(df["tgt_id"].to_numpy())
        if with_label:
            ys.append(df["label"].to_numpy(dtype=np.int8))
    X = np.concatenate(xs)
    s1 = np.concatenate(s1s)
    tgt = np.concatenate(tgts)
    y = np.concatenate(ys) if with_label else None
    return X, y, s1, tgt


def train(feat_dir, model_out, params=None, num_round=2000, val_frac=0.1, seed=42):
    """Train the LightGBM matcher on labeled features; early-stop on a held slice.

    The validation slice is split BY S1 ENTITY (not by pair) so a pair and its
    sibling candidates never straddle the split - avoids optimistic early stopping.
    """
    import lightgbm as lgb

    X, y, s1, _tgt = _load_feature_shards(feat_dir, with_label=True)
    print("train matrix: %s  positives: %d (%.2f%%)"
          % (X.shape, int(y.sum()), 100.0 * y.mean()), file=sys.stderr)

    rng = np.random.default_rng(seed)
    uniq_s1 = np.unique(s1)
    val_s1 = set(rng.choice(uniq_s1, size=max(1, int(len(uniq_s1) * val_frac)),
                            replace=False).tolist())
    is_val = np.fromiter((x in val_s1 for x in s1), dtype=bool, count=len(s1))

    train_set = lgb.Dataset(X[~is_val], label=y[~is_val],
                            feature_name=FEATURE_NAMES)
    val_set = lgb.Dataset(X[is_val], label=y[is_val], reference=train_set)

    if params is None:
        pos = max(1, int(y[~is_val].sum()))
        neg = int((~is_val).sum()) - pos
        params = {
            "objective": "binary",
            "metric": "average_precision",
            "num_leaves": 255,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "min_data_in_leaf": 50,
            "scale_pos_weight": neg / pos,   # counter the ~1:8 class imbalance
            "verbosity": -1,
            "num_threads": 0,                # use all cores
        }

    model = lgb.train(
        params, train_set, num_boost_round=num_round,
        valid_sets=[val_set], valid_names=["val"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )
    os.makedirs(os.path.dirname(os.path.abspath(model_out)), exist_ok=True)
    model.save_model(model_out)
    print("saved model -> %s (best_iter=%s)" % (model_out, model.best_iteration),
          file=sys.stderr)

    # Report feature importance for the methodology doc / audit.
    imp = sorted(zip(FEATURE_NAMES, model.feature_importance(importance_type="gain")),
                 key=lambda kv: kv[1], reverse=True)
    print("top features by gain:", file=sys.stderr)
    for name, g in imp[:10]:
        print("  %-22s %.0f" % (name, g), file=sys.stderr)
    return model


def score(feat_dir, model_path, out_dir):
    """Score every candidate pair; write (s1_id, tgt_id, score) parquet shards."""
    import lightgbm as lgb
    model = lgb.Booster(model_file=model_path)
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(feat_dir, "*.parquet")))
    for si, f in enumerate(files):
        df = pd.read_parquet(f)
        X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
        s = model.predict(X, num_iteration=model.best_iteration)
        out = pd.DataFrame({"s1_id": df["s1_id"], "tgt_id": df["tgt_id"],
                            "score": s.astype(np.float32)})
        out.to_parquet(os.path.join(out_dir, "score_%05d.parquet" % si), index=False)
    print("scored %d shard(s) -> %s" % (len(files), out_dir), file=sys.stderr)
    return out_dir
