# TASKS

_Updated 26 Sep 2026, 00:15._

## 1. Where we are

| | |
|---|---|
| **Leaderboard score** | **0.56** |
| Predict-all-empty floor | 0.0559 |
| Measured blocking recall | 0.5708 (ceiling ~0.885) |
| Submission | `output/matching_results.tsv`, 1,732,544 rows, validator PASS incl. ID check |

## 2. Workspace cleaned and consolidated

Cut from **~50 files to 22**, with the pipeline itself going from 8 modules to 5.

| Was | Now |
|---|---|
| `textnorm.py` + `v3feat.py` + `v3block.py` | `core.py` + `block.py` |
| `train.py` + `score.py` | `train.py` (`--score-pred`) |
| `probe_recall.py` + `analyze_score.py` | `measure.py` (`recall` / `curve`) |
| `predict.py` | `predict.py` (unchanged) |
| `full_v3.ps1` + `v4_density.ps1` + `sweep.ps1` | `tools/run.ps1` (`-Mode`) |
| `src/baseline/_archive/` (18 files) | deleted; negative result recorded in `docs/FINAL_REPORT.md` |
| 3 stale model files, 10 scratch cache files (2.1 GB), ~80 empty logs | deleted |

**Merge verified, not assumed:** `block.py` was run against the previous
`v3block.py` on the same inputs and the two outputs are **byte-identical**
(md5 `b309853b15c41c3993dddfebc6be581e`, 10,994,105 bytes each). Merging did not
change the pipeline's behaviour.

Kiro's files (`src/*.py` top level, `pipeline.py`, `run_camber.py`,
`artifacts/`, `camber_*.zip`) were left untouched.

## 3. The defect that cost us points

Run 1 tuned its threshold on training candidates built from **30%** of the
Source-2/3 rows and served it on candidates built from **100%**:

| | pairs | S1 entities | candidates/entity |
|---|---:|---:|---:|
| training | 44,928,982 | 2,206,821 | 20.4 |
| test | 53,405,476 | 1,732,544 | **30.8** |

A threshold that is safe against 20.4 candidates is too permissive against 30.8,
and a false merge on a true singleton entity costs a full 1.0. That is the most
likely reason the leaderboard reads 0.56 rather than higher.

## 4. In progress - density-matched rebuild

`tools/run.ps1` is running with **no `--limit` on either split**, so the
threshold is tuned at the density it is served at.

| Stage | Work | State |
|---|---|---|
| 1. Block train, full density | 2.21M S1 x 10.32M rows | **RUNNING** (started 00:11) |
| 2. Train + re-tune threshold | at matched density | queued |
| 3. Block test | **reused** - already full density, saves 93 min | skipped |
| 4. Predict | both output files | queued |
| 5. Validate | official checker | queued |

Track it with `Get-Content logs\run.log -Wait`.

**Decision rule: ship the new submission only if its validation score at matched
density beats 0.4565.** A newer model is not automatically a better one.

## 5. Next, in order

1. **Density-matched retrain** (running) - calibration fix, cheap.
2. **Raise blocking recall above 0.57** - a second pass restricted to Source-1 entities that produced few candidates, using character n-gram and sorted-token-pair keys. Probe with `-Mode recall` first; only run it full if the probe shows a clear gain. Recall 0.70 implies a ceiling near 0.93.
3. **Features** - token edit distance, phonetic codes, address normalisation. Only worth it once blocking is not the binding constraint.

## 6. Measured negative result, kept in the report not the code

A per-entity adaptive decision rule was built and measured against the global
threshold on 60,000 held-out entities: **0.4498 vs 0.4562**. It lost at every
`m_prior` tried, because `train.py` already tunes the global threshold against
the exact metric. Recorded in `docs/FINAL_REPORT.md` section 5.4 so it is not
re-explored.

## 7. Rules kept

- No external data, geocoding, registries or APIs
- `country` is an open string set - France never filtered
- All 1,732,544 test entities in the submission
- Nothing over ~500 MB loaded whole into RAM
