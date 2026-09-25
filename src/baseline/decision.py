#!/usr/bin/env python3
"""Per-entity ADAPTIVE decision rule, replacing the single global threshold.

Why this exists
---------------
The competition metric is not a plain F-beta, and a single global threshold is
provably the wrong shape of decision rule for it.  Per entity:

    m == 0 and k == 0  ->  1.0        correct singleton
    m == 0 and k  > 0  ->  0.0        false merge
    otherwise          ->  1.25 * t / (0.25 * m + k)

Two facts fall out of that formula, and predict.py's global threshold honours
neither of them:

1. An entity with **no** true match is worth 1.0 for free just by predicting
   nothing.  Every prediction made on such an entity destroys a full point.
2. An entity that *does* have matches scores ``1.25t / (0.25m + k)``, which
   *rises* with recall.  Being timid is penalised too.

So the optimal number of predictions per entity is not a constant.  It depends
on how many candidates that entity has and how strong they look.

The rule implemented here
-------------------------
For an entity with candidates whose calibrated probabilities are p_1..p_k
(sorted descending):

    M    = sum(p_i)                     expected number of true matches
    P0   = prod(1 - p_i)                P(no candidate is a true match)
    EV(S)= (1 - P0) * 1.25 * sum_{i in S} p_i / (0.25 * M + |S|)

    EV(empty) = P0                        <- the singleton payoff

We evaluate every prefix of the descending-probability order and keep the best.
That is exact for this objective: any optimal S is a prefix, because the
formula depends on S only through |S| and sum of its p_i, and swapping a
included low-probability candidate for an excluded higher one always helps.

Cost is O(k log k) per entity, k <= a few hundred.

This module is import-only; `decide.py` is the CLI.
"""

from __future__ import annotations

import math


def expected_score(probs, m_prior: float = 1.0) -> tuple:
    """Pick the best prefix of `probs` (descending) under the real metric.

    Parameters
    ----------
    probs : sequence of float
        Calibrated match probabilities for one Source-1 entity's candidates.
        Order does not matter; they are sorted internally.
    m_prior : float
        Multiplier on the expected match count M.  Values slightly above 1.0
        make the rule a little more willing to predict; 1.0 is the unbiased
        estimate.  Tuned on held-out training entities (see `decide.py`).

    Returns
    -------
    (k, ev, baseline_ev)
        k        - how many candidates to emit (0 = predict a singleton)
        ev       - expected macro-F0.5 contribution of that entity
        baseline_ev - expected score of always predicting nothing
    """
    p = sorted((float(x) for x in probs if x > 0.0), reverse=True)
    if not p:
        return 0, 1.0, 1.0

    m = m_prior * sum(p)
    # P(none of the candidates is real)
    log_p0 = 0.0
    for x in p:
        log_p0 += math.log1p(-min(x, 1.0 - 1e-9))
    p0 = math.exp(log_p0)

    denom_const = 0.25 * m
    if denom_const <= 0.0:
        return 0, p0, p0

    weight = 1.0 - p0
    best_k, best_ev, cum = 0, p0, 0.0
    ev = p0
    for i, x in enumerate(p, start=1):
        cum += x
        ev = weight * 1.25 * cum / (denom_const + i)
        if ev > best_ev:
            best_ev, best_k = ev, i
    return best_k, best_ev, p0


def best_prefix_mask(probs, m_prior: float = 1.0) -> int:
    """How many of the top-scoring candidates to keep for one entity."""
    return expected_score(probs, m_prior)[0]