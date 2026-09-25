#!/usr/bin/env python3
"""Stage 1: candidate generation (blocking) with a per-S1 coverage floor.

Goal: for every S1 entity, produce a small set of likely S2/S3 matches, cheaply,
without ever materializing the ~17e12 naive pair space.

Method - IDF-weighted inverted-index retrieval, FORWARD-primary (S1 -> targets):
  * Build an inverted index token -> [target rows], per country, over S2+S3.
  * For each S1 entity, gather targets that share tokens, score by summed IDF
    (rare shared tokens count more), and keep the top-K.
  * FORWARD-primary guarantees every S1 gets up to K candidates (red-team D1): we
    never let target-side ranking decide whether an S1 is reachable.

Why forward-primary and not reverse-primary: the metric is macro-averaged PER S1
entity. Reverse (target->owner) top-k can leave a low-match S1 with zero candidates,
scoring it 0 - and those low-match entities dominate the macro loss. Forward top-K
gives every S1 a floor of candidates. (Reverse rank is added later as a feature.)

Memory safety (Camber SMALL, 64 GB, but also smaller nodes):
  * The token->postings index is the one big structure. Postings are stored as
    arrays of int32 row-ids. We cap per-token postings (drop ultra-common tokens,
    like plan's df prune) so no single token explodes memory.
  * Targets are read in chunks; only compact arrays (ids, token-id lists) are kept.
  * S1 entities are queried in batches; candidates are streamed to disk (parquet),
    never all held at once.

Country gating: index and query per country. NOTE (red-team D17): this is a hard
gate justified by the observed 100% country agreement on train links. It is applied
as a retrieval optimization; the matcher still sees a country-agreement feature. If a
cross-country link ever mattered it would be unreachable - a known, accepted risk for
this dataset.

Output: candidate_pairs parquet shards with columns (s1_id, tgt_id) and, for reuse,
the tokenized fields cached to disk so Stage 2 doesn't re-tokenize.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from array import array
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import (  # noqa: E402
    address_tokens, name_tokens, trade_name_variants, dominant_script,
)

CHUNK = 200_000


# --------------------------------------------------------------------------- records

def _prefix(entity_id: str) -> str:
    """Source prefix from an entity id: S1-/S2-/S3-."""
    return entity_id.split("-", 1)[0] if "-" in entity_id else ""


def iter_source(path: str):
    """Yield chunks of a source TSV as DataFrames (streaming)."""
    return pd.read_csv(path, sep="\t", chunksize=CHUNK, dtype=str,
                       keep_default_na=False, na_filter=False, encoding="utf-8")


# --------------------------------------------------------------- vocabulary + df pass

def build_vocab_and_df(target_paths: list[str], min_len: int = 2):
    """First pass over targets: assign each token an int id and count document freq.

    Returns (token_to_id, df_array) where df_array[token_id] = document frequency.
    Tokens are name tokens + address tokens combined (union retrieval, plan C.9).
    """
    token_to_id: dict[str, int] = {}
    df_counts: list[int] = []

    def tid(tok: str) -> int:
        i = token_to_id.get(tok)
        if i is None:
            i = len(df_counts)
            token_to_id[tok] = i
            df_counts.append(0)
        return i

    for path in target_paths:
        print("  vocab pass: %s" % os.path.basename(path), file=sys.stderr)
        for chunk in iter_source(path):
            for name, addr in zip(chunk["business_name"], chunk["business_address"]):
                toks = set(name_tokens(name)) | set(address_tokens(addr))
                for t in toks:
                    if len(t) >= min_len:
                        df_counts[tid(t)] += 1
    return token_to_id, np.asarray(df_counts, dtype=np.int64)


def compute_idf(df_array: np.ndarray, n_docs: int) -> np.ndarray:
    """Standard smoothed IDF: log((N+1)/(df+1)) + 1."""
    return np.log((n_docs + 1.0) / (df_array + 1.0)) + 1.0


# ------------------------------------------------------------------- inverted index

class InvertedIndex:
    """token_id -> postings (int32 target row ids), with a df cap to bound memory.

    Postings are built as python arrays then frozen into numpy arrays so retrieval
    can score with vectorized ops instead of per-posting python loops.
    """

    def __init__(self, n_tokens: int, df_cap: int):
        self.postings: list[array] = [array("i") for _ in range(n_tokens)]
        self.np_postings: list[np.ndarray] | None = None
        self.df_cap = df_cap
        self.target_ids: list[str] = []          # row -> tgt entity id
        self.target_tok: list[np.ndarray] = []    # row -> array of token ids
        self.target_country: list[str] = []
        self.target_source: list[str] = []
        self.country_codes: np.ndarray | None = None   # row -> int country code
        self._country_to_code: dict[str, int] = {}

    def add_target(self, tgt_id, country, source, token_ids: np.ndarray):
        row = len(self.target_ids)
        self.target_ids.append(tgt_id)
        self.target_tok.append(token_ids)
        self.target_country.append(country)
        self.target_source.append(source)
        for t in token_ids:
            p = self.postings[t]
            if len(p) < self.df_cap:      # skip ultra-common tokens past the cap
                p.append(row)
        return row

    def freeze(self):
        """Convert postings to numpy and build an int country-code array for masking."""
        self.np_postings = [np.frombuffer(p, dtype=np.int32) if len(p) else
                            np.empty(0, dtype=np.int32) for p in self.postings]
        self.postings = []  # free the python arrays
        codes = []
        for c in self.target_country:
            code = self._country_to_code.get(c)
            if code is None:
                code = len(self._country_to_code)
                self._country_to_code[c] = code
            codes.append(code)
        self.country_codes = np.asarray(codes, dtype=np.int32)

    def country_code(self, country: str) -> int:
        return self._country_to_code.get(country, -1)


def build_index(target_paths, token_to_id, df_cap, min_len=2) -> InvertedIndex:
    """Second pass: build the inverted index and cache per-target token arrays."""
    idx = InvertedIndex(len(token_to_id), df_cap)
    for path in target_paths:
        print("  index pass: %s" % os.path.basename(path), file=sys.stderr)
        for chunk in iter_source(path):
            for eid, country, name, addr in zip(
                chunk["entity_id"], chunk["country"],
                chunk["business_name"], chunk["business_address"],
            ):
                toks = set(name_tokens(name)) | set(address_tokens(addr))
                ids = [token_to_id[t] for t in toks
                       if len(t) >= min_len and t in token_to_id]
                idx.add_target(eid, country, _prefix(eid),
                               np.asarray(sorted(set(ids)), dtype=np.int32))
    return idx


# ------------------------------------------------------------------------- retrieval

def query_s1(name, addr, token_to_id, idf, idx: InvertedIndex, country_code,
             top_k, min_len=2):
    """Return up to top_k (tgt_row, score) for one S1 entity, country-gated.

    Score = sum of IDF over shared tokens (a fast cosine-like accumulator),
    computed with numpy: concatenate the postings of all query tokens, weight each
    by the token's IDF, then sum per target row via np.add.at. We also expand the
    name via trade-name variants so a wrapped fake name still retrieves.
    """
    query_tokens = set()
    for variant in trade_name_variants(name):
        query_tokens.update(variant.split())
    query_tokens.update(address_tokens(addr))
    q_ids = [token_to_id[t] for t in query_tokens
             if len(t) >= min_len and t in token_to_id]
    if not q_ids:
        return []

    rows_parts, weight_parts = [], []
    for t in q_ids:
        post = idx.np_postings[t]
        if post.size:
            rows_parts.append(post)
            weight_parts.append(np.full(post.size, idf[t], dtype=np.float32))
    if not rows_parts:
        return []

    rows = np.concatenate(rows_parts)
    weights = np.concatenate(weight_parts)

    # Country gate via boolean mask on the int country-code array.
    keep = idx.country_codes[rows] == country_code
    if not keep.any():
        return []
    rows, weights = rows[keep], weights[keep]

    # Unique target rows + summed IDF weight per row.
    uniq, inverse = np.unique(rows, return_inverse=True)
    acc = np.zeros(uniq.size, dtype=np.float32)
    np.add.at(acc, inverse, weights)

    if uniq.size > top_k:
        top = np.argpartition(acc, -top_k)[-top_k:]
        uniq, acc = uniq[top], acc[top]
    order = np.argsort(acc)[::-1]
    return list(zip(uniq[order].tolist(), acc[order].tolist()))


def generate_candidates(s1_path, target_paths, out_dir, top_k=50, df_cap=40_000,
                        min_len=2):
    """Full Stage-1 run: build index over targets, retrieve top-K per S1, write shards.

    Returns the path to the written candidate parquet directory and the index (so the
    caller can reuse cached tokenizations for feature building).
    """
    os.makedirs(out_dir, exist_ok=True)

    print("Stage 1: building vocabulary + document frequencies ...", file=sys.stderr)
    token_to_id, df_array = build_vocab_and_df(target_paths, min_len)

    # N (document count) for IDF = total number of target rows.
    n_targets = 0
    for p in target_paths:
        with open(p, "rb") as fh:
            next(fh, None)
            for _ in fh:
                n_targets += 1
    idf = compute_idf(df_array, n_targets)
    print("  vocab: %d tokens | targets: %d" % (len(token_to_id), n_targets),
          file=sys.stderr)

    print("Stage 1: building inverted index ...", file=sys.stderr)
    idx = build_index(target_paths, token_to_id, df_cap, min_len)
    idx.freeze()

    print("Stage 1: retrieving top-%d candidates per S1 ..." % top_k, file=sys.stderr)
    shard_id = 0
    n_s1 = 0
    buf_s1, buf_tgt, buf_rank, buf_score = [], [], [], []

    def flush():
        nonlocal shard_id, buf_s1, buf_tgt, buf_rank, buf_score
        if not buf_s1:
            return
        df = pd.DataFrame({
            "s1_id": buf_s1, "tgt_id": buf_tgt,
            "rank": buf_rank, "block_score": buf_score,
        })
        df.to_parquet(os.path.join(out_dir, "cand_%05d.parquet" % shard_id),
                      index=False)
        shard_id += 1
        buf_s1, buf_tgt, buf_rank, buf_score = [], [], [], []

    for chunk in iter_source(s1_path):
        for eid, country, name, addr in zip(
            chunk["entity_id"], chunk["country"],
            chunk["business_name"], chunk["business_address"],
        ):
            n_s1 += 1
            cc = idx.country_code(country)
            items = query_s1(name, addr, token_to_id, idf, idx, cc, top_k, min_len)
            for rank, (row, score) in enumerate(items):
                buf_s1.append(eid)
                buf_tgt.append(idx.target_ids[row])
                buf_rank.append(rank)
                buf_score.append(float(score))
            # Coverage floor: even if items is empty, the S1 still appears in the
            # final results (pipeline guarantees a row); here we simply record none.
        if len(buf_s1) >= 2_000_000:
            flush()
    flush()
    print("  retrieved candidates for %d S1 entities into %d shard(s)"
          % (n_s1, shard_id), file=sys.stderr)
    return out_dir, idx, token_to_id, idf


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 1 blocking / candidate generation.")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--out", default="artifacts/candidates")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--df-cap", type=int, default=40_000)
    args = ap.parse_args()

    base = os.path.join(args.data_dir, args.split)
    s1 = os.path.join(base, "%s_source1.tsv" % args.split)
    targets = [os.path.join(base, "%s_source2.tsv" % args.split),
               os.path.join(base, "%s_source3.tsv" % args.split)]
    generate_candidates(s1, targets, args.out, args.top_k, args.df_cap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
