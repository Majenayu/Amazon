#!/usr/bin/env python3
"""Shared, label-free text normalization for the entity-resolution pipeline.

Every transformation here is derived only from the text itself - no external
databases, no hand-typed lookup tables of world knowledge. Where a map is needed
(state codes, street abbreviations), it is either (a) a purely typographic rule
that any auditor can see is not "external business data", or (b) built at runtime
from co-occurrence inside the provided files (see build_maps.py). This module holds
only the typographic/structural rules; the co-occurrence maps are passed in.

Design notes tied to the empirical findings:

* Accent folding is applied to LATIN script only. Indic combining marks are vowels
  and must be preserved (finding C.7 / plan S1.2).
* No discriminative token is deleted. Generic tokens (center, service, group) are
  kept and left for IDF to downweight, because ~48% of names collide (C.4 / S1.3).
* Address is treated as an unordered bag of comma-separated components (C.5 / S1.7).
* Numeric view strips leading zeros but keeps qualifiers like 458-a, bis, ter.
* Domains/handles are decomposed into tokens (C.5 / S1.6).

Stdlib + unicodedata only, so it runs anywhere with no heavy deps.
"""

from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------- script

# Unicode block ranges for the scripts that appear in the data (C.7). Used only to
# decide whether accent-folding is safe (Latin) and to emit a coarse script flag.
_SCRIPT_RANGES = [
    ("DEVANAGARI", 0x0900, 0x097F),
    ("BENGALI", 0x0980, 0x09FF),
    ("GURMUKHI", 0x0A00, 0x0A7F),
    ("GUJARATI", 0x0A80, 0x0AFF),
    ("ORIYA", 0x0B00, 0x0B7F),
    ("TAMIL", 0x0B80, 0x0BFF),
    ("TELUGU", 0x0C00, 0x0C7F),
    ("KANNADA", 0x0C80, 0x0CFF),
    ("MALAYALAM", 0x0D00, 0x0D7F),
]


def dominant_script(text: str) -> str:
    """Return a coarse script label for the string: LATIN or an Indic block name.

    Chooses the script with the most characters. Digits/punctuation/space are
    ignored. Empty or purely non-letter strings return LATIN (the safe default for
    accent handling).
    """
    counts: dict[str, int] = {}
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        label = "LATIN"
        for name, lo, hi in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                label = name
                break
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return "LATIN"
    return max(counts, key=counts.get)


def has_native_script(text: str) -> bool:
    """True if the string contains any non-Latin (Indic) letter."""
    for ch in text:
        cp = ord(ch)
        for _name, lo, hi in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                return True
    return False


# ----------------------------------------------------------------------- accent fold

def fold_latin_accents(text: str) -> str:
    """Strip combining marks from LATIN letters only; preserve Indic vowels.

    We NFKD-decompose, then drop combining marks that follow a Latin base letter,
    while keeping combining marks that belong to Indic scripts (they are phonemic).
    """
    out = []
    decomposed = unicodedata.normalize("NFKD", text)
    prev_is_latin = False
    for ch in decomposed:
        if unicodedata.combining(ch):
            # Drop the mark only if the base it attaches to was Latin.
            if prev_is_latin:
                continue
            out.append(ch)
            continue
        cp = ord(ch)
        is_latin = (0x41 <= cp <= 0x7A) or (0xC0 <= cp <= 0x24F)
        prev_is_latin = is_latin
        out.append(ch)
    return "".join(out)


# ------------------------------------------------------------------------- tokenize

_APOS = "\u2019\u2018'`"                       # curly + straight apostrophes
_TOKEN_SPLIT = re.compile(r"[^0-9a-z\u0900-\u0d7f]+")  # keep latin, digits, Indic
_DOMAIN_RE = re.compile(r"([a-z0-9-]+)\.(com|net|org|co|in|io|biz|info|us|fr)\b")
_HANDLE_RE = re.compile(r"[@#]([a-z0-9_]+)")


def _standardize_ampersand(text: str) -> str:
    return text.replace("&", " and ")


def _decompose_domains_handles(text: str) -> str:
    """Turn domains/handles into their word tokens so they can match plain names.

    northernsilver.com -> northernsilver ; @smithveterinary -> smithveterinary
    The domain suffix (com/net/...) is dropped; it carries no business identity.
    """
    text = _DOMAIN_RE.sub(lambda m: " " + m.group(1).replace("-", " ") + " ", text)
    text = _HANDLE_RE.sub(lambda m: " " + m.group(1) + " ", text)
    return text


def basic_clean(text: str) -> str:
    """Lowercase, fold Latin accents, standardize apostrophes/ampersands.

    Returns a cleaned string still containing spaces/punctuation; tokenization is a
    separate step so callers can choose name vs address handling.
    """
    if not text:
        return ""
    text = text.strip()
    text = fold_latin_accents(text)
    text = text.lower()
    for a in _APOS:
        text = text.replace(a, "")
    text = _standardize_ampersand(text)
    text = _decompose_domains_handles(text)
    return text


def tokenize(text: str) -> list[str]:
    """Split cleaned text into tokens (latin words, Indic runs, digit runs)."""
    if not text:
        return []
    cleaned = basic_clean(text)
    return [t for t in _TOKEN_SPLIT.split(cleaned) if t]


# --------------------------------------------------------------------------- numeric

_LEADING_ZERO = re.compile(r"^0+(\d)")


def normalize_number(tok: str) -> str:
    """Strip leading zeros but keep the token's qualifier structure.

    00203 -> 203 ; 458-a -> 458-a ; 1/2 -> 1/2 ; bis/ter left untouched by caller.
    Only the pure numeric prefix is de-zeroed; the rest is preserved.
    """
    m = re.match(r"^(\d+)(.*)$", tok)
    if not m:
        return tok
    num, rest = m.group(1), m.group(2)
    num = num.lstrip("0") or "0"
    return num + rest


def numeric_tokens(text: str) -> list[str]:
    """Return the set of numeric tokens in an address, de-zeroed.

    House numbers are a strong-but-noisy signal (C.5); we surface them separately so
    the matcher can compare them directly.
    """
    out = []
    for tok in tokenize(text):
        if any(c.isdigit() for c in tok):
            out.append(normalize_number(tok))
    return out


# --------------------------------------------------------- name / address token sets

def name_tokens(name: str) -> list[str]:
    """Tokenize a business name. Keeps every token (no suffix deletion)."""
    return tokenize(name)


def address_tokens(address: str) -> list[str]:
    """Tokenize an address as an unordered bag; de-zero numeric tokens in place.

    S3 reorders whole addresses (C.5), so order carries no reliable signal; a bag of
    tokens is the robust representation.
    """
    out = []
    for tok in tokenize(address):
        out.append(normalize_number(tok) if any(c.isdigit() for c in tok) else tok)
    return out


# ----------------------------------------------------------------- trade-name split

# Wrappers that prepend/insert a FAKE name (C.5 / S1.5). Split on these and keep BOTH
# sides as name hypotheses; the caller decides how to weight them.
_TRADE_SPLIT = re.compile(r"\b(?:dba|d/b/a|t/a|t\\a|f/k/a|fka|a/k/a|aka)\b")


def trade_name_variants(name: str) -> list[str]:
    """Return candidate name strings from a possible trade-name wrapper.

    "Deltasol dba Ajmer Lining" -> ["deltasol", "ajmer lining", "<full>"]. Always
    includes the whole cleaned string as a fallback hypothesis.
    """
    cleaned = " ".join(basic_clean(name).split())  # collapse internal whitespace
    parts = [" ".join(p.split()) for p in _TRADE_SPLIT.split(cleaned) if p.strip()]
    variants = list(dict.fromkeys([p for p in parts if p] + [cleaned]))  # dedup
    return [v for v in variants if v]


# -------------------------------------------------------------------- state canonic.

def canonical_state(last_component: str, state_map: dict[str, str] | None) -> str:
    """Canonicalize the trailing address component (state/region) via a co-occurrence
    map passed in from build_maps.py.

    state_map maps any observed surface form (2-letter code, full name, native
    script) to a shared canonical key. If no map or no hit, return the cleaned form
    so at least exact-string agreement still works.
    """
    key = " ".join(tokenize(last_component))
    if state_map and key in state_map:
        return state_map[key]
    return key


if __name__ == "__main__":
    # Tiny self-check with representative noisy strings from the findings.
    samples = [
        ("Mr Bhagyashree India Private Limited (ID: 10652)", "name"),
        ("Deltasol dba Ajmer Lining Private Limited", "name"),
        ("northernsilver.com", "name"),
        ("गुजरात बिजनेस प्राइवेट लिमिटेड", "name"),
        ("00203, Rue d' Amsterdam, BORDEAUX, Gironde", "addr"),
        ("D.no.b324b-1-55a, KA", "addr"),
    ]
    for s, kind in samples:
        if kind == "name":
            print(repr(s))
            print("   tokens :", name_tokens(s))
            print("   variants:", trade_name_variants(s))
            print("   script :", dominant_script(s))
        else:
            print(repr(s))
            print("   tokens :", address_tokens(s))
            print("   numbers:", numeric_tokens(s))
