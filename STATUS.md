# STATUS - Amazon ML Challenge (Cline's track)

_Last updated 25 Sep 2026, 23:52._

## COMPLETE - submission built and validated

| | |
|---|---|
| **Validation macro F0.5** | **0.4565** (floor 0.0559 = 8.2x) |
| **Official validator** | **PASS - safe to submit** |
| `output/matching_results.tsv` | 1,732,544 rows, 1,423,133 non-empty, 309,411 empty |
| `output/candidate_pairs.tsv` | 1,732,544 rows, 1,587,779 with candidates |
| Test candidate pairs scored | 53,405,476 |
| Pairs above threshold 0.8170 | 4,783,978 |

All 5 pipeline stages ran on the full 2.52 GB (26,435,994 rows).

## Stage timings

| Stage | Result | Time |
|---|---|---|
| 1. Block train | 44,928,982 pairs, 4.37 GB, 2,400,928 true matches | 72 min |
| 2. Train model | 5.32M rows, threshold 0.8170, F0.5 0.4565 | 6.8 min |
| 3. Block test | 53,405,476 pairs, 4.85 GB | 93 min |
| 4. Predict | both files written | 10 min |
| 5. Validate | PASS | 8 min |

## Measured position

| | Value | Source |
|---|---|---|
| Predict-all-empty floor | 0.0559 | `models/v3_meta.json` |
| **Validation macro F0.5** | **0.4565** | `models/v3_meta.json` |
| Blocking recall ceiling | 0.5708 | `logs/sweep.log` |
| Implied best possible score | ~0.885 | `1.25r/(0.25+r)` at r=0.5708 |

0.4565 is measured on 441,611 held-out training entities at full candidate
density. The leaderboard number will differ: test blocking used all 9.97M source
rows while validation used 30%, and France has no training labels. It is the
best current estimate, not a promise.

## Four bugs found and fixed during the run

1. **False stage failure.** `full_v3.ps1` printed `STAGE 1/5 FAIL exit=` right after stage 1 had *successfully* written 4.37 GB, and aborted. Cause: PowerShell returns `$null` for `Process.ExitCode` unless the native handle is cached first. Fixed via `$p.Handle`.
2. **No resume.** Added `StageCached` to skip stages whose output exists. Saved 72 min then 93 min on relaunches.
3. **`decide.py` missing numpy import** - `NameError` crashed stage 4. Fixed.
4. **`predict.py` logging bug** - `{:.0f}` with a `%` tuple raised `TypeError` on the last log line, *after* both output files were fully written, which made the orchestrator report a false failure. Fixed to `%.0f`.

## Measured negative result

An adaptive per-entity decision rule was built on the theory that the metric
rewards singletons for predicting nothing *and* rewards recall for entities
with matches, so a single global threshold should lose. Measured on 60,000
held-out entities:

| Rule | macro F0.5 |
|---|---|
| best global threshold (t=0.8571) | **0.4562** |
| adaptive, best of 5 m_prior values | 0.4498 |

It loses everywhere, so stage 4 ships the global threshold. Code kept as a
recorded result, not deleted and not claimed as an improvement.

## Corrections to earlier claims

I previously said 0.99 was impossible, then later implied this pipeline would
reach the 0.90+ tier. Neither was supported when I said it. Measured position:
**0.4565 achieved, ceiling ~0.885** given 0.5708 blocking recall.

## Files

| Path | What |
|---|---|
| `output/matching_results.tsv` | **the submission** (80 MB) |
| `output/candidate_pairs.tsv` | candidate set (678 MB) |
| `models/v3.joblib`, `v3_meta.json` | model, threshold, F0.5, full sweep curve |
| `data/cache/v3_train.tsv` | 44.9M labelled train pairs (4.37 GB) |
| `data/cache/v3_test.tsv` | 53.4M test pairs (4.85 GB) |
| `docs/FINAL_REPORT.md` | full technical report |
| `logs/full_run.log` | one line per stage |

## Next improvement, ranked

Raise blocking recall above 0.57 - a second pass over only the Source-1 entities
with few candidates, using character n-gram keys. Recall 0.70 implies a ceiling
near 0.93. Worth more than any model change.
