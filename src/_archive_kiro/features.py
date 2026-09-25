#!/usr/bin/env python3
"""Stage 2a: turn each candidate (s1_id, tgt_id) pair into numeric features.

Features are computed from the normalized text of both sides. They are grounded in
the empirical signal analysis (C.3): address/name token overlap, exact-name match,
numeric agreement, state agreement, script flags, and rarity/IDF signals.

Memory safety: we first load compact per-record dictionaries
  s1_id  -> (name_tokens, addr_tokens, num_tokens, state_key, script, raw_name)
  tgt_id -> same
built by streaming the source files in chunks. Only the fields we need are kept, as
python tuples of small lists, which fits comfortably in 64 GB for the full data
(records are short: name ~25 chars, addr ~50). Candidate pairs are then read shard by
shard and features written back as parquet, so the pair-feature matrix never lives in
RAM all at once.

The token->IDF weight is passed in from blocking (label-free, computed over sources).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import (  # noqa: E402
    address_tokens, name_tokens, numeric_tokens, dominant_script,
    canonical_state, basic_clean,
)

CHUNK = 200_000


# ------------------------------------------------------------------- record loading

def load_records(path, state_map=None):
    """Stream a source file into a compact dict keyed by entity_id.

    Value tuple: (name_tok_set, addr_tok_set, num_tok_set, state_key, script,
                  name_str, addr_empty)
    """
    records = {}
    reader = pd.read_csv(path, sep="\t", chunksize=CHUNK, dtype=str,
                         keep_default_na=False, na_filter=False, encoding="utf-8")
    for chunk in reader:
        for eid, name, addr in zip(
            chunk["entity_id"], chunk["business_name"], chunk["business_address"],
        ):
            nt = frozenset(name_tokens(name))
            at = frozenset(address_tokens(addr))
            num = frozenset(numeric_tokens(addr))
            last = addr.split(",")[-1] if addr.strip() else ""
            state = canonical_state(last, state_map) if last else ""
            records[eid] = (
                nt, at, num, state, dominant_script(name),
                " ".join(basic_clean(name).split()), not addr.strip(),
            )
    return records


# ------------------------------------------------------------------- set similarity

def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _containment(a: frozenset, b: frozenset) -> float:
    """Overlap / size of the smaller set (robust when one side is truncated)."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _dice(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    return 2.0 * len(a & b) / (len(a) + len(b)) if (a or b) else 0.0


def _max_shared_idf(a: frozenset, b: frozenset, token_to_id, idf) -> float:
    """Highest IDF among shared tokens - a rare shared token is strong evidence."""
    shared = a & b
    if not shared:
        return 0.0
    best = 0.0
    for t in shared:
        i = token_to_id.get(t)
        if i is not None and idf[i] > best:
            best = float(idf[i])
    return best


# ----------------------------------------------------------------------- pair feats

FEATURE_NAMES = [
    "name_jaccard", "name_containment", "name_dice", "name_exact",
    "addr_jaccard", "addr_containment", "addr_dice",
    "num_overlap", "num_contradict",
    "state_agree",
    "name_token_set_ratio", "name_partial_ratio",
    "addr_token_set_ratio",
    "script_match", "either_native",
    "tgt_addr_empty",
    "max_shared_name_idf", "max_shared_addr_idf",
    "n_shared_name", "n_shared_addr",
    "block_score", "block_rank",
    "is_s3",
]


def pair_features(s1rec, tgtrec, token_to_id, idf, block_score, block_rank):
    """Compute the feature vector for one candidate pair."""
    (s1_nt, s1_at, s1_num, s1_state, s1_script, s1_name, _s1_empty) = s1rec
    (t_nt, t_at, t_num, t_state, t_script, t_name, t_empty) = tgtrec

    name_jac = _jaccard(s1_nt, t_nt)
    addr_jac = _jaccard(s1_at, t_at)

    num_inter = len(s1_num & t_num)
    num_union = len(s1_num | t_num)
    num_overlap = num_inter / num_union if num_union else 0.0
    # contradiction: both have numbers but none shared (corrupted house number, C.5)
    num_contradict = 1.0 if (s1_num and t_num and num_inter == 0) else 0.0

    state_agree = 1.0 if (s1_state and t_state and s1_state == t_state) else 0.0
    script_match = 1.0 if s1_script == t_script else 0.0
    either_native = 1.0 if (s1_script != "LATIN" or t_script != "LATIN") else 0.0

    return [
        name_jac, _containment(s1_nt, t_nt), _dice(s1_nt, t_nt),
        1.0 if (s1_name and s1_name == t_name) else 0.0,
        addr_jac, _containment(s1_at, t_at), _dice(s1_at, t_at),
        num_overlap, num_contradict,
        state_agree,
        fuzz.token_set_ratio(s1_name, t_name) / 100.0,
        fuzz.partial_ratio(s1_name, t_name) / 100.0,
        fuzz.token_set_ratio(" ".join(sorted(s1_at)), " ".join(sorted(t_at))) / 100.0,
        script_match, either_native,
        1.0 if t_empty else 0.0,
        _max_shared_idf(s1_nt, t_nt, token_to_id, idf),
        _max_shared_idf(s1_at, t_at, token_to_id, idf),
        float(len(s1_nt & t_nt)), float(len(s1_at & t_at)),
        float(block_score), float(block_rank),
        0.0,  # is_s3: set by build_features from the tgt_id prefix
    ]


def build_features(cand_dir, s1_records, tgt_records, token_to_id, idf,
                   out_dir, gt_map=None):
    """Read candidate shards, emit feature shards. If gt_map given, add a label col.

    gt_map: {s1_id: set(true_tgt_ids)} for training; None at inference.
    """
    import glob
    os.makedirs(out_dir, exist_ok=True)
    shard_files = sorted(glob.glob(os.path.join(cand_dir, "*.parquet")))
    for si, f in enumerate(shard_files):
        cand = pd.read_parquet(f)
        rows = []
        labels = []
        keep_s1, keep_tgt = [], []
        for s1_id, tgt_id, brank, bscore in zip(
            cand["s1_id"], cand["tgt_id"], cand["rank"], cand["block_score"],
        ):
            s1rec = s1_records.get(s1_id)
            tgtrec = tgt_records.get(tgt_id)
            if s1rec is None or tgtrec is None:
                continue
            feats = pair_features(s1rec, tgtrec, token_to_id, idf, bscore, brank)
            feats[-1] = 1.0 if tgt_id.startswith("S3-") else 0.0  # is_s3
            rows.append(feats)
            keep_s1.append(s1_id)
            keep_tgt.append(tgt_id)
            if gt_map is not None:
                labels.append(1 if tgt_id in gt_map.get(s1_id, ()) else 0)

        if not rows:
            continue
        arr = np.asarray(rows, dtype=np.float32)
        out = pd.DataFrame(arr, columns=FEATURE_NAMES)
        out.insert(0, "s1_id", keep_s1)
        out.insert(1, "tgt_id", keep_tgt)
        if gt_map is not None:
            out["label"] = np.asarray(labels, dtype=np.int8)
        out.to_parquet(os.path.join(out_dir, "feat_%05d.parquet" % si), index=False)
        print("  features shard %d: %d pairs" % (si, len(out)), file=sys.stderr)
    return out_dir


def load_gt_map(path):
    """Load ground truth into {s1_id: set(tgt_id)} (empty set for singletons)."""
    gt = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False)
    out = {}
    for s1, mids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        out[s1] = {x for x in mids.split(",") if x} if mids.strip() else set()
    return out
