"""Text normalisation, blocking keys and the 20 similarity features.

Shared by block.py (to build candidate pairs) and by nothing else, because
every other stage reads the feature columns straight out of the pairs file.
Merged from the old textnorm.py + v3feat.py on 26 Sep 2026.

Everything here is deliberately cheap: the dataset is 26.4M rows and this
module runs over all of it.
"""

from __future__ import annotations

import math
import re

# Legal suffixes are stripped from the END of a name only. Dropping them in the
# middle would destroy names like "Limited Edition Games".
LEGAL_SUFFIXES = frozenset("""
inc incorporated llc ltd limited corp corporation co company plc gmbh pvt pte
private limited liability partners llp srl sarl sa bv nv kg ag oy ab as pty
holdings holding group
""".split())

_NON_ALNUM = re.compile(r"[^0-9a-z\u0900-\u097f ]+")
_PIN = re.compile(r"(?<!\d)(\d{5,6})(?!\d)")
NUMERIC = set("0123456789")

TOK_MIN = 4     # tokens shorter than this are too common to key on
PRE_MIN = 4     # token-prefix key length
ADDR_MIN = 5     # address tokens shorter than this are too common to key on

# Generic address words that would create huge posting lists but carry almost
# no identity. They are skipped at key-build time (the tok-cap trim is the
# second line of defence). Deliberately small: over-skipping kills the
# transliteration recall this key type exists for.
ADDR_SKIP = frozenset("""
road street avenue lane nagar colony block floor building shop plot near near
opposite main cross layout area sector phase extension ext road street
floor ground first second
""".split())


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------

def normalize_name(s: str) -> str:
    """lower-case, keep latin + devanagari, drop punctuation, strip legal suffix."""
    if not s:
        return ""
    s = _NON_ALNUM.sub(" ", s.lower())
    toks = s.split()
    while toks and toks[-1] in LEGAL_SUFFIXES:
        toks.pop()
    return " ".join(toks)


def tokens(name: str) -> set:
    return set(name.split()) if name else set()


def trigrams(s: str) -> set:
    s = s.replace(" ", "_")
    if len(s) < 3:
        return {s} if s else set()
    return {s[i:i + 3] for i in range(len(s) - 2)}


def pins(address: str) -> tuple:
    """US ZIP (5, sometimes 5-4) and India PIN (6) as plain digit strings."""
    if not address:
        return ()
    seen = []
    for m in _PIN.findall(address):
        if m not in seen:
            seen.append(m)
        if len(seen) == 3:
            break
    return tuple(seen)


def cc(c: str) -> str:
    """Country as a plain lower-case string. Never a whitelist: France has no
    training labels and must not be filtered out."""
    return (c or "").strip().lower()


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / float(len(a) + len(b) - inter)


# --------------------------------------------------------------------------
# blocking keys
# --------------------------------------------------------------------------

def key_sig(n: str) -> str:
    """Sorted-token signature: survives word-order swaps."""
    return "|".join(sorted(n.split()))


def addr_tokens(address: str) -> set:
    """ normalised address tokens worth blocking on.

    Lower-cased, split on whitespace, keep alnum tokens of length >= ADDR_MIN
    that are not in ADDR_SKIP and not purely numeric (street numbers are
    covered by the PIN / numeric-token features, and bare numbers like
    '12' would collide across millions of rows).
    """
    if not address:
        return set()
    out = set()
    for t in address.lower().split():
        # strip surrounding punctuation: "road," -> "road"
        t = t.strip(".,;:()[]{}'\"-")
        if len(t) < ADDR_MIN:
            continue
        if t in ADDR_SKIP:
            continue
        if t.isdigit():
            continue
        # must contain a letter to avoid pure codes colliding
        if not any(ch.isalpha() for ch in t):
            continue
        out.add(t)
    return out


def keys_for(n: str, c: str, pn, atok=None) -> list:
    """(weight, key) pairs for one record.

    E exact name and S sorted signature are near-certain matches, T/Z are
    solid evidence, A (address token) is weaker per-token but is the ONLY
    signal for transliteration pairs where the name shares nothing, X
    (4-char token prefix) is only a typo hint. The weight orders which
    posting lists are visited first and how candidates are ranked.

    The union of these six key types is what lifts blocking recall. A pair
    becomes a candidate if ANY key links it.
    """
    out = []
    if n:
        out.append((8, "E|" + c + "|" + n))
        out.append((6, "S|" + c + "|" + key_sig(n)))
    seen_t, seen_x = set(), set()
    for t in n.split():
        if len(t) >= TOK_MIN and t not in seen_t:
            seen_t.add(t)
            out.append((3, "T|" + c + "|" + t))
        if len(t) >= PRE_MIN and t[:PRE_MIN] not in seen_x:
            seen_x.add(t[:PRE_MIN])
            out.append((1, "X|" + c + "|" + t[:PRE_MIN]))
    if atok:
        for t in atok:
            # 1.5 each: two shared address tokens (3.0) pass the default
            # gate on address evidence alone, one token alone does not.
            out.append((1.5, "A|" + c + "|" + t))
    for z in pn:
        out.append((3, "Z|" + c + "|" + z))
    return out


# --------------------------------------------------------------------------
# the 20 features
# --------------------------------------------------------------------------
# All are language-agnostic, so France (15% of test, zero training labels) is
# handled by the same code path as every other country.

FEATURE_NAMES = [
    "name_exact",     #  0  normalised names identical
    "name_jacc",      #  1  token Jaccard
    "name_tri",       #  2  character 3-gram Jaccard
    "name_pre3",      #  3  first three characters equal
    "addr_jacc",      #  4  address token Jaccard
    "pin_eq",         #  5  postal code shared
    "country_eq",     #  6  same country label
    "len_ratio",      #  7  name length ratio
    "tok_contain",    #  8  |A n B| / min(|A|,|B|) containment, not symmetric
    "sig_eq",         #  9  sorted token signature equal (word-order swap)
    "first_tok_eq",   # 10  first token equal
    "last_tok_eq",    # 11  last token equal
    "lcp_ratio",      # 12  longest common prefix / longest name
    "num_tok_jacc",   # 13  numeric token Jaccard (street / unit numbers)
    "addr_num_eq",    # 14  any numeric address token shared
    "sub_str",        # 15  one normalised name contains the other
    "min_len_log",    # 16  log1p(min length) / 5, clipped
    "addr_len_ratio", # 17  address length ratio
    "tok_min_log",    # 18  min token count / 6, clipped
    "tok_pre4_jacc",  # 19  Jaccard of per-token 4-char prefixes (typo tolerant)
]


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


def features(r1: dict, r2: dict) -> list:
    """Similarity features for one candidate pair.

    Both records are dicts carrying: n (normalised name), atok (address token
    set), country (raw label), pins (tuple of digit strings).
    """
    n1, n2 = r1["n"], r2["n"]
    t1, t2 = tokens(n1), tokens(n2)

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


HDR = "s1_id\tother_id\t" + "\t".join(FEATURE_NAMES) + "\tlabel"
SCORE_HDR = "s1_id\tother_id\tevidence\tlabel"


def make_record(entity_id: str, name: str, address: str, country: str) -> dict:
    """Light record: keeps only what the features and keys need.

    Raw strings are deliberately not retained - Source-1 alone is 2.2M rows and
    holding the raw text costs about 1 GB.
    """
    return {
        "id": entity_id,
        "n": normalize_name(name),
        "atok": set((address or "").lower().split()),
        "country": country or "",
        "pins": pins(address),
    }
