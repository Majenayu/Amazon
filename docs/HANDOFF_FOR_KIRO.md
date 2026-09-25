# Handoff note — ARCHIVED 26 Sep 2026

Kiro's parallel pipeline (`src/_archive_kiro/`) is archived by owner request.
The only active pipeline is `src/baseline/` (5 modules) run via
`tools/run.ps1`. Nothing in the archive is imported by the active pipeline
(verified: no `baseline` imports reference it, no archive file imports
`baseline`). Restore with `git mv src/_archive_kiro/<file> <dest>` if needed.

## Facts that are already verified (do not re-derive)

Measured by `src/explore.py`, full data (`notes/eda_summary.txt`):

- Files are **tab separated**. `pd.read_csv(..., sep="\t")`. Columns:
  `entity_id`, `business_name`, `business_address`, `country`.
- Row counts — train: S1 2,206,821 / S2 5,034,616 / S3 5,285,603 / GT 2,206,821.
  test: S1 1,732,544 / S2 4,887,273 / S3 5,082,316. **Total 26.4M rows / 2.4 GB.**
- Ground truth: 94.42% of S1 have >=1 match, **5.58% are singletons** (no match).
  Average 3.67 matches per matched entity. Total true matches 7,638,365.
- **Predicting "empty" for every entity scores 0.0558 macro F_0.5.** That is the floor.
- Test contains **France (~15%)**, absent from train. `country` is an open set of
  strings — never hard-code `{US, India}`, never one-hot encode a fixed country list.
- Empty addresses: S1 = 0%, S2/S3 = ~3.3%. Never assume an address exists.
- Test country mix: India 47%, US 38%, France 14%. Train: US 60%, India 40%.

## Metric (macro F_0.5, computed per S1 entity then averaged)

```
m = |true matches|, k = |predicted|, t = |correct|
if m == 0 and k == 0 : 1.0
if m == 0 and k >  0 : 0.0
else                 : 1.25 * t / (0.25 * m + k)
```

Precision counts twice as much as recall: a wrong merge hurts more than a miss.
Singletons are scored too — predicting a match for them is an instant 0.0.

## Hard rules

- Every S1 row in `test_source1.tsv` must appear in `matching_results.tsv`
  (empty second column is valid and scores 1.0 for singletons).
- `candidate_pairs.tsv` column is `candidate_entity_ids`, and final
  `matching_entity` predictions must be a subset of the candidates.
- No external data (no geocoding, no business registries, no APIs) — DQ.
- Do not load a 500 MB file whole into RAM (only 5.4 GB free on this machine).
  Stream it or read in `chunksize=` chunks.
- Run `python src/validate_submission.py --matching output/matching_results.tsv
  --candidate output/candidate_pairs.tsv --test-dir data/test` before uploading.

## Small sample for fast iteration

`data/sample/` = 11.8 MB with **correct labels** (10,032 labelled S1).
Rebuild with `python src/make_sample.py --n-s1 2000` for a tiny pasteable version.
