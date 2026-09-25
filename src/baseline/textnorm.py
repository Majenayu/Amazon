"""Text normalisation + blocking keys shared by the baseline pipeline.

Everything here is deliberately cheap: the full dataset is 26M rows and this
module runs several times over it.
"""

from __future__ import annotations

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


def normalize_name(s: str) -> str:
    """lowercase, keep latin + devanagari, drop punctuation, strip legal suffix."""
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


def block_keys(name: str, address: str, country: str) -> tuple:
    """Return (normalised_name, [keys]) for one record.

    Keys are prefixed so different key types can never collide:
      P|country|name[:3]   first three characters of the cleaned name
      T|country|token      first few tokens of length >= 4 (any script)
      Z|country|pin        postal code
    """
    n = normalize_name(name)
    c = (country or "").strip().lower()
    keys = []
    if len(n) >= 3:
        keys.append("P|%s|%s" % (c, n[:3]))
    for t in n.split()[:6]:
        if len(t) >= 4:
            keys.append("T|%s|%s" % (c, t))
    for p in pins(address):
        keys.append("Z|%s|%s" % (c, p))
    return n, keys


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / float(len(a) + len(b) - inter)


def features(rec1: dict, rec2: dict) -> list:
    """Similarity features for a candidate pair.

    rec1 is the Source-1 record, rec2 the Source-2/3 record. Both must already
    carry the keys produced by block_keys(): n (normalised name), raw name,
    address, country.
    """
    n1, n2 = rec1["n"], rec2["n"]
    t1, t2 = tokens(n1), tokens(n2)
    g1, g2 = trigrams(n1), trigrams(n2)

    p1, p2 = set(rec1.get("pins") or ()), set(rec2.get("pins") or ())
    pin_eq = 1.0 if (p1 and p2 and (p1 & p2)) else 0.0

    len1, len2 = len(n1), len(n2)
    longest = max(len1, len2)
    len_ratio = (min(len1, len2) / float(longest)) if longest else 1.0

    a1, a2 = rec1.get("atok") or set(), rec2.get("atok") or set()

    return [
        1.0 if n1 == n2 and n1 else 0.0,      # exact match after cleaning
        jaccard(t1, t2),                       # name token overlap
        jaccard(g1, g2),                       # name character 3-gram overlap
        1.0 if (len1 >= 3 and n1[:3] == n2[:3]) else 0.0,
        jaccard(a1, a2),                       # address token overlap
        pin_eq,                                # postal code equality
        1.0 if (rec1.get("country") or "").lower() == (rec2.get("country") or "").lower() else 0.0,
        len_ratio,
    ]


FEATURE_NAMES = [
    "name_exact", "name_jacc", "name_tri", "name_pre3",
    "addr_jacc", "pin_eq", "country_eq", "len_ratio",
]


def make_record(entity_id: str, name: str, address: str, country: str) -> dict:
    n, keys = block_keys(name, address, country)
    # NOTE: raw name/address are intentionally NOT kept - on the full dataset
    # Source-1 alone is 2.2M rows and holding the raw strings costs ~1 GB.
    return {
        "id": entity_id,
        "n": n,
        "ntok": set(n.split()) if n else set(),
        "keys": keys,
        "atok": set((address or "").lower().split()),
        "country": country or "",
        "pins": pins(address),
    }
