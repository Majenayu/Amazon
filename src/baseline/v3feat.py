"""v3 features: 20 similarity signals for a (Source-1, Source-2/3) pair.

Design notes
------------
* every feature is a plain float in [0, 1] except the length ones, which are
  squashed so tree models can split them cleanly;
* features are computed from the *normalised* name plus the raw address, and
  none of them depend on language, so they still work for France, which has no
  training labels at all;
* the list is longer than v1 (8 features) because the hard cases here are typos,
  word-order swaps and abbreviations, and each of those needs its own signal.

FEATURE_NAMES_V3 order is the order emitted by features_v3().
"""

from __future__ import annotations

import math

from baseline.textnorm import jaccard, normalize_name, pins, trigrams

FEATURE_NAMES_V3 = [
    "name_exact",     # 0  normalised names identical
    "name_jacc",      # 1  token Jaccard
    "name_tri",       # 2  character 3-gram Jaccard
    "name_pre3",      # 3  first three characters equal
    "addr_jacc",      # 4  address token Jaccard
    "pin_eq",         # 5  postal code shared
    "country_eq",     # 6  same country label
    "len_ratio",      # 7  name length ratio
    "tok_contain",    # 8  |A n B| / min(|A|,|B|)  containment, not symmetric
    "sig_eq",         # 9  sorted token signature equal (word-order swap)
    "first_tok_eq",   # 10 first token equal
    "last_tok_eq",    # 11 last token equal
    "lcp_ratio",      # 12 longest common prefix / longest name
    "num_tok_jacc",   # 13 numeric token Jaccard (street / unit numbers)
    "addr_num_eq",    # 14 any numeric address token shared
    "sub_str",        # 15 one normalised name contains the other
    "min_len_log",    # 16 log1p(min length) / 5, clipped
    "addr_len_ratio",  # 17 address length ratio
    "tok_min_log",    # 18 min token count / 6, clipped
    "tok_pre4_jacc",  # 19 Jaccard of per-token 4-char prefixes (typo tolerant)
]

NUMERIC = set("0123456789")


def tok_prefixes(name: str, k: int = 4) -> set:
    return {t[:k] for t in name.split() if t}


def numeric_tokens(s: str) -> set:
    return {t for t in (s or "").split() if t and t[0] in NUMERIC}


def lcp(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def features_v3(r1, r2) -> list:
    """r1 = Source-1 record, r2 = Source-2/3 record.

    Both are dicts with: n (normalised name), atok (address tokens set),
    country (raw label), pins (tuple of digit strings).
    """
    n1, n2 = r1["n"], r2["n"]
    t1 = set(n1.split()) if n1 else set()
    t2 = set(n2.split()) if n2 else set()

    inter = len(t1 & t2)
    contain = inter / float(min(len(t1), len(t2))) if (t1 and t2) else 0.0

    p1 = set(r1.get("pins") or ())
    p2 = set(r2.get("pins") or ())
    pin_eq = 1.0 if (p1 and p2 and (p1 & p2)) else 0.0

    l1, l2 = len(n1), len(n2)
    longest = max(l1, l2)
    len_ratio = (min(l1, l2) / float(longest)) if longest else 1.0

    a1 = r1.get("atok") or set()
    a2 = r2.get("atok") or set()
    a_long = max(len(a1), len(a2))
    addr_len_ratio = (min(len(a1), len(a2)) / float(a_long)) if a_long else 1.0

    n1_num, n2_num = numeric_tokens(n1), numeric_tokens(n2)
    a1_num, a2_num = numeric_tokens(" ".join(a1)), numeric_tokens(" ".join(a2))

    return [
        1.0 if (n1 == n2 and n1) else 0.0,
        jaccard(t1, t2),
        jaccard(trigrams(n1), trigrams(n2)),
        1.0 if (l1 >= 3 and n1[:3] == n2[:3]) else 0.0,
        jaccard(a1, a2),
        pin_eq,
        1.0 if (r1.get("country") or "").lower() == (r2.get("country") or "").lower() else 0.0,
        len_ratio,
        contain,
        1.0 if (t1 and t1 == t2 and sorted(t1) == sorted(t2)) else 0.0,
        1.0 if (t1 and t2 and n1.split()[0] == n2.split()[0]) else 0.0,
        1.0 if (t1 and t2 and n1.split()[-1] == n2.split()[-1]) else 0.0,
        (lcp(n1, n2) / float(longest)) if longest else 1.0,
        jaccard(n1_num, n2_num),
        1.0 if (a1_num and a2_num and (a1_num & a2_num)) else 0.0,
        1.0 if (n1 and n2 and (n1 in n2 or n2 in n1)) else 0.0,
        min(1.0, math.log1p(min(l1, l2)) / 5.0) if (n1 or n2) else 0.0,
        addr_len_ratio,
        min(1.0, min(len(t1), len(t2)) / 6.0) if (t1 or t2) else 0.0,
        jaccard(tok_prefixes(n1), tok_prefixes(n2)),
    ]


def make_record_v3(entity_id: str, name: str, address: str, country: str) -> dict:
    """Light record: keeps only what the v3 features and keys need."""
    n = normalize_name(name)
    return {
        "id": entity_id,
        "n": n,
        "atok": set((address or "").lower().split()),
        "country": country or "",
        "pins": pins(address),
    }