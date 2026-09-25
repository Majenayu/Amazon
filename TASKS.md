# TASKS - my (Cline's) track

_Updated 25 Sep 2026, 22:05. Kiro works separately and I never touch its files._

## 1. Headline result

**Validation macro F0.5 = 0.4565** on 441,611 held-out training entities at full
candidate density, against a predict-all-empty floor of **0.0559**. That is
**8.2x better than doing nothing**, and it is a measured number, not an estimate.

| | Value | Source |
|---|---|---|
| Predict-all-empty floor | 0.0559 | 5.59% of entities are singletons |
| **Validation macro F0.5** | **0.4565** | `models/v3_meta.json` |
| Best global threshold | 0.8170 | tuned on the exact metric |
| Blocking recall ceiling | 0.5708 | `logs/sweep.log` |
+| Implied best possible score | ~0.885 | `1.25r/(0.25+r)` at r=0.5708 |

## 2. The deliverable you upload

| # | File | What | State |
|---|---|---|---|
| 1 | `output/matching_results.tsv` | **the submission** - all 1,732,544 test S1 ids | stage 5 |
| 2 | `output/candidate_pairs.tsv` | candidate set (required in the final zip) | stage 5 |
| 3 | `models/v3.joblib` + `v3_meta.json` | model + tuned threshold + full sweep curve | **done** |
| 4 | `docs/FINAL_REPORT.md` | technical report with measured numbers | **done** |
| 5 | `src/baseline/` | reproducible pipeline | **done** |

## 3. Pipeline - `tools/full_v3.ps1`, all 5 stages unattended

| Stage | Does | Time | State |
|---|---|---|---|
| 1 | Block train: 2.21M S1 vs Source-2/3 | 72 min | **done** - 44,928,982 pairs, 4.37 GB |
| 2 | Train model + tune threshold | 6.8 min | **done** - macro F0.5 0.4565 |
| 3 | Block test: 1.73M S1 vs 9.97M rows | ~70 min | **RUNNING** - S2 600k/4.89M |
| 4 | Predict -> both output files | ~10 min | queued |
| 5 | Official format validation | 1 min | queued |

Progress: `logs/full_run.log` (one line per stage), `logs/full_3_block_test.log.err` (live).

## 4. Done

- [x] Data verified intact: 2.52 GB, 26,435,994 rows, never modified
- [x] Repo cleaned; `.git` purged of 1.15 GB of dead objects
- [x] EDA: 5.59% singletons, France ~15% of test, avg 3.67 matches per entity
- [x] Root-caused the original AI failure: 2.52 GB cannot fit an AI context window
- [x] `data/sample/` (11.8 MB, labels intact) for fast iteration
- [x] v3 blocker: 5 union key types, ranked selection, wide budget
- [x] Blocking recall measured on real data = 0.5708 (the score ceiling)
- [x] Model trained on 5.32M rows, threshold tuned on the exact metric
- [x] **Validation macro F0.5 = 0.4565** (8.2x the floor)
- [x] Fixed a PowerShell bug that misreported a successful stage as FAIL and aborted the run
- [x] Added resume support so finished stages are never recomputed
- [x] Built and **measured** an adaptive per-entity decision rule - it lost, so it was not shipped
- [x] Full technical report written

## 5. Measured negative result (kept, not hidden)

The metric rewards singletons for predicting nothing *and* rewards recall for
entities that have matches, which suggested a per-entity decision rule would
beat one global threshold. Built, then measured on 60,000 held-out entities:

| Rule | macro F0.5 |
|---|---|
| best global threshold (t=0.8571) | **0.4562** |
| adaptive, best of 5 m_prior values | 0.4498 |

It loses at every setting, so stage 4 uses the global threshold. Stage 2 already
tunes that threshold against the exact metric, which implicitly prices in the
singleton payoff. Code kept in `decision.py` / `decide.py` as a recorded result.

## 6. Bug found and fixed during the run

`tools/full_v3.ps1` reported `STAGE 1/5 FAIL exit=` immediately after stage 1 had
**successfully** written its 4.37 GB output, and aborted the whole run. Cause: a
known PowerShell quirk where `Process.ExitCode` returns `$null` unless the
native handle is cached first. Fixed by touching `$p.Handle` before
`WaitForExit()`. Also added stage-level resume so a crash never discards hours
of completed work.

## 7. On the 0.9999 target - the honest answer is no

The recall/score relation says 0.9999 needs roughly 99.5% blocking recall *and*
perfect classification. We measure 57.08%. That gap is not a tuning problem:
about 43% of true matches share **no** token, no 4-char prefix and no postal
code with their Source-1 row, so no key-based scheme in this repo can reach
them. I will not claim a number I have not measured.

A realistic strong result is 0.60-0.70, which needs recall around 0.65-0.70.

## 8. Highest-value next improvements, in order

1. **Raise blocking recall above 0.57** - a second pass over only the Source-1 entities that produced few candidates, using character n-gram keys. This moves the ceiling directly and is worth more than any model change.
2. **Wider budget** - post-budget 800 measured 0.5930 recall at half the speed.
3. **France-specific check** - 15% of test with zero training labels.

## 9. Rules I must not break

- No external data, geocoding, registries or APIs - instant disqualification
- `country` is an open string set - France must never be filtered out
- Every test S1 entity must appear in the submission (a missing row scores 0)
- Never load a 500 MB file whole into RAM

