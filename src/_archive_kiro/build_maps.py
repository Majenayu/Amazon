#!/usr/bin/env python3
"""Derive canonicalization maps ONLY from in-file co-occurrence (no external data).

Why this matters (rules + red-team D16): the competition forbids external lookup
tables. Any state-code<->name or abbreviation map must be discovered from the
provided files. Critically, we build these maps WITHOUT using ground-truth links, so
the exact same procedure works for France (which has no labels) and cannot be accused
of leaking the label generator. Every map entry keeps a support count for the audit.

How the state map is built - purely structural, label-free:
  The last comma-component of an address is the state/region/departement (C.5).
  Different sources encode the same state differently (code vs full name vs native
  script). We link surface forms that share the SAME set of city co-occurrences: if
  "tx" and "texas" both appear as the trailing component for the same cities (the
  component just before them), they denote the same state. We cluster surface forms
  by the cities they co-occur with; each cluster gets one canonical key.

This is unsupervised (uses only the source files, never train_ground_truth). It runs
per country so US "tx"/"texas" never merges with anything Indian.

Output: a JSON file mapping surface_form -> canonical_key, plus support counts, for
each country. Loaded by the pipeline at feature time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import tokenize  # noqa: E402

CHUNK = 200_000


def _last_two_components(address: str) -> tuple[str, str] | None:
    """Return (city_component, state_component) as cleaned token strings.

    The address is comma-separated; the trailing component is the state-like field
    and the one before it is the city-like field. Returns None if we can't get two.
    """
    parts = [p.strip() for p in address.split(",") if p.strip()]
    if len(parts) < 2:
        return None
    city = " ".join(tokenize(parts[-2]))
    state = " ".join(tokenize(parts[-1]))
    if not city or not state:
        return None
    return city, state


def _accumulate(path: str, per_country_city_state: dict):
    """Stream a source file and record, per country, which cities each state-surface
    co-occurs with, and how often each surface form appears."""
    reader = pd.read_csv(path, sep="\t", chunksize=CHUNK, dtype=str,
                         keep_default_na=False, na_filter=False, encoding="utf-8")
    for chunk in reader:
        for country, addr in zip(chunk["country"], chunk["business_address"]):
            if not addr:
                continue
            got = _last_two_components(addr)
            if not got:
                continue
            city, state = got
            store = per_country_city_state.setdefault(country, {})
            entry = store.setdefault(state, {"support": 0, "cities": defaultdict(int)})
            entry["support"] += 1
            entry["cities"][city] += 1


def _cluster_states(store: dict, min_shared: int = 2) -> dict[str, str]:
    """Cluster state-surface forms that share enough city co-occurrences.

    Union-find over surface forms: two surfaces merge if they share at least
    `min_shared` distinct cities. The canonical key of a cluster is its
    highest-support surface form (stable, human-readable for the audit).
    """
    surfaces = list(store.keys())
    parent = {s: s for s in surfaces}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Only real states cluster: a genuine state/region has many addresses, so we only
    # treat high-support surface forms as clustering candidates. This drops the long
    # tail of noisy trailing components (typos, stray tokens) that would otherwise
    # create an O(n^2) explosion of surface-form pairs. min_support scales with data.
    supports = sorted((store[s]["support"] for s in surfaces), reverse=True)
    # Keep at most the top ~200 surface forms per country (states/regions are few),
    # and require a floor of support so rare junk never enters clustering.
    min_support = max(min_shared, supports[min(len(supports) - 1, 200)] if supports else 0)
    candidates = [s for s in surfaces if store[s]["support"] >= min_support]

    # Build city -> candidate-surfaces index, capping per city to avoid cliques.
    city_to_surfaces = defaultdict(set)
    cand_set = set(candidates)
    for s in candidates:
        for city in store[s]["cities"]:
            bucket = city_to_surfaces[city]
            if len(bucket) < 50:            # cap surfaces per city
                bucket.add(s)

    shared_count = defaultdict(int)
    for city, surfs in city_to_surfaces.items():
        surfs = sorted(surfs)
        for i in range(len(surfs)):
            for j in range(i + 1, len(surfs)):
                shared_count[(surfs[i], surfs[j])] += 1

    for (a, b), n in shared_count.items():
        if n >= min_shared:
            union(a, b)

    # Canonical key per cluster = highest-support member.
    clusters = defaultdict(list)
    for s in surfaces:
        clusters[find(s)].append(s)
    mapping = {}
    for members in clusters.values():
        canon = max(members, key=lambda s: store[s]["support"])
        for s in members:
            mapping[s] = canon
    return mapping


def build(data_dirs: list[str], out_path: str, min_shared: int) -> None:
    per_country_city_state: dict = {}
    files = []
    for d in data_dirs:
        for name in ("source1.tsv", "source2.tsv", "source3.tsv"):
            for prefix in ("train_", "test_"):
                p = os.path.join(d, prefix + name)
                if os.path.isfile(p):
                    files.append(p)
    if not files:
        print("No source files found under: %s" % data_dirs, file=sys.stderr)
        sys.exit(1)

    for p in files:
        print("scanning %s ..." % os.path.basename(p), file=sys.stderr)
        _accumulate(p, per_country_city_state)

    result = {}
    for country, store in per_country_city_state.items():
        mapping = _cluster_states(store, min_shared=min_shared)
        # Keep support counts for the audit trail.
        support = {s: store[s]["support"] for s in store}
        result[country] = {"map": mapping, "support": support}
        n_clusters = len(set(mapping.values()))
        print("  %s: %d surface forms -> %d canonical states"
              % (country, len(mapping), n_clusters), file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False)
    print("saved -> %s" % out_path, file=sys.stderr)


def load_state_maps(path: str) -> dict[str, dict[str, str]]:
    """Load {country: {surface_form: canonical_key}} for use in the pipeline."""
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return {country: blob["map"] for country, blob in raw.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Build label-free canonicalization maps.")
    ap.add_argument("--data-dir", default="data", help="folder with train/ and test/")
    ap.add_argument("--out", default="artifacts/state_maps.json")
    ap.add_argument("--min-shared", type=int, default=2,
                    help="min shared cities to merge two state surfaces (default 2)")
    args = ap.parse_args()

    dirs = [os.path.join(args.data_dir, "train"), os.path.join(args.data_dir, "test")]
    build(dirs, args.out, args.min_shared)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
