#!/usr/bin/env python3
"""Stage 3: decision layer. Turn pair scores into the final matched_entity_ids.

The metric is macro F_0.5 (precision weighted 2x). Two facts drive the design:
  * Every S1 entity must appear exactly once, even France and singletons (empty list).
  * One target has one owner (C.1): a target should be claimed by at most one S1.

Acceptance (red-team D6/D10/D13): rather than three stacked heuristics, we accept per
entity by a calibrated absolute threshold on the model score, with a set-size-aware
refinement: we always allow the single best candidate to be accepted at a lower bar,
because for a 1-match entity a miss scores 0 while surviving an FP is cheaper. This
is a pragmatic stand-in for full expected-F_0.5 maximization and is tuned on holdout.

One-owner (D23): after per-entity acceptance we resolve target contention globally -
if two S1 entities accept the same target, the higher score keeps it. This is a clean
per-target argmax, stated explicitly so it is reproducible.

Every S1 in the required set is emitted; entities with no accepted candidate get an
empty list (correct for true singletons, and the safe action when all candidates are
weak).
"""

from __future__ import annotations

import glob
import os
import sys

import pandas as pd


def load_scores(score_dir):
    """Return {s1_id: [(tgt_id, score), ...]} sorted by score desc."""
    by_s1 = {}
    for f in sorted(glob.glob(os.path.join(score_dir, "*.parquet"))):
        df = pd.read_parquet(f)
        for s1, tgt, sc in zip(df["s1_id"], df["tgt_id"], df["score"]):
            by_s1.setdefault(s1, []).append((tgt, float(sc)))
    for s1 in by_s1:
        by_s1[s1].sort(key=lambda kv: kv[1], reverse=True)
    return by_s1


def decide(by_s1, required_s1, threshold=0.5, best_floor=0.30, max_matches=11):
    """Produce {s1_id: [tgt_id, ...]} for every required S1, honoring one-owner.

    threshold   : accept a candidate if score >= threshold.
    best_floor  : additionally accept the single top candidate if its score >=
                  best_floor (protects 1-match entities where a miss costs a full 1.0).
    max_matches : safety cap flag (observed max 11 in train); we cap accepted set size.
    """
    # Pass 1: per-entity provisional acceptance.
    provisional = {}
    for s1 in required_s1:
        cands = by_s1.get(s1, [])
        accepted = [(t, sc) for t, sc in cands if sc >= threshold]
        if not accepted and cands and cands[0][1] >= best_floor:
            accepted = [cands[0]]
        accepted = accepted[:max_matches]
        provisional[s1] = accepted

    # Pass 2: one-owner contention. Each target kept by its highest-scoring claimer.
    best_claim = {}   # tgt -> (score, s1)
    for s1, accepted in provisional.items():
        for tgt, sc in accepted:
            cur = best_claim.get(tgt)
            if cur is None or sc > cur[0]:
                best_claim[tgt] = (sc, s1)

    final = {}
    for s1 in required_s1:
        kept = [tgt for tgt, sc in provisional[s1]
                if best_claim.get(tgt, (None, None))[1] == s1]
        final[s1] = kept
    return final


def write_results(final, required_s1, out_path):
    """Write matching_results.tsv: one row per required S1, empty when no match."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for s1 in required_s1:
            ids = final.get(s1, [])
            fh.write("%s\t%s\n" % (s1, ",".join(ids)))
    print("wrote %d rows -> %s" % (len(required_s1), out_path), file=sys.stderr)


def write_candidates(by_s1, required_s1, out_path):
    """Write candidate_pairs.tsv: the last blocking stage fed to the matcher."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1 in required_s1:
            cands = [t for t, _ in by_s1.get(s1, [])]
            fh.write("%s\t%s\n" % (s1, ",".join(cands)))
    print("wrote %d candidate rows -> %s" % (len(required_s1), out_path),
          file=sys.stderr)


def read_required_s1(s1_path):
    """Every S1 entity id in the (test) source1 file - the required output rows."""
    ids = []
    with open(s1_path, encoding="utf-8") as fh:
        next(fh, None)
        for line in fh:
            if line.strip():
                ids.append(line.split("\t", 1)[0].strip())
    return ids
