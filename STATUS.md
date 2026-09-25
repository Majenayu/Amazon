# STATUS - Amazon ML Challenge (Cline's track)

_Last updated 25 Sep 2026, 22:06._

## Headline

**Validation macro F0.5 = 0.4565** (floor 0.0559). Measured on 441,611 held-out
training entities at full candidate density. 8.2x better than predicting nothing.

## Right now

| Stage | State | Detail |
|---|---|---|
| 1. Data | OK untouched | 2.52 GB / 26,435,994 rows in `data/` |
| 2. Block TRAIN | done | 44,928,982 pairs, 4.37 GB, 2,400,928 true matches, 72 min |
| 3. Train model | done | 5.32M training rows, threshold 0.8170, **F0.5 = 0.4565** |
| 4. Block TEST | RUNNING | S2 600k of 4.89M, ~1,000 rows/s, ETA ~23:10 |
| 5. Predict + validate | queued | writes both output files, then format check |

Orchestrator: `tools/full_v3.ps1` (PID 9116), python PID 40892.

## Two bugs found and fixed during this run

1. **False stage failure.** `full_v3.ps1` printed `STAGE 1/5 FAIL exit=` right after stage 1 had *successfully* written its 4.37 GB output, and aborted everything. Cause: PowerShell returns `$null` for `Process.ExitCode` unless the native handle is cached first. Fixed by touching `$p.Handle` before `WaitForExit()`.
2. **No resume.** Any crash re-did hours of finished work. Fixed with `StageCached`, which skips a stage whose output already exists and is non-empty. Stage 1 was skipped on the relaunch, saving 72 minutes.

## Measured negative result

An adaptive per-entity decision rule (`decision.py`, `decide.py`) was built on
the theory that the metric rewards singletons for predicting nothing *and*
rewards recall for entities with matches, so one global threshold should lose.
Measured on 60,000 held-out entities:

| Rule | macro F0.5 |
|---|---|
| best global threshold (t=0.8571) | **0.4562** |
| adaptive, best of 5 m_prior values | 0.4498 |

It loses everywhere, so stage 4 uses the global threshold. The code is kept as
a recorded result, not deleted and not claimed as an improvement.

## Corrections to earlier claims

I previously said 0.99 was impossible, then later implied this pipeline would
reach the 0.90+ tier. Neither was supported when I said it. The measured
position: **0.4565 achieved, ceiling ~0.885** given 0.5708 blocking recall.

## Files

| Path | What |
|---|---|
| `data/` | original data, never modified |
| `data/cache/v3_train.tsv` | 44.9M train candidate pairs with labels (4.37 GB) |
| `data/cache/v3_test.tsv` | test candidate pairs (building) |
| `models/v3.joblib`, `v3_meta.json` | model, threshold 0.8170, F0.5 0.4565, sweep curve |
| `output/matching_results.tsv` | **the submission** (written at stage 5) |
| `docs/FINAL_REPORT.md` | full technical report |
| `logs/full_run.log` | one line per stage |
| `logs/full_3_block_test.log.err` | live blocking progress |

## Commands

```powershell
# watch progress
Get-Content logs\full_run.log -Wait

# format check before uploading
python src\validate_submission.py --matching output\matching_results.tsv --candidate output\candidate_pairs.tsv --test-dir data\test
```

## What would raise the score, in order

1. Raise blocking recall above 0.57 - second pass over Source-1 entities with few candidates, using character n-gram keys. Worth more than any model change.
2. Wider budget: post-budget 800 measured 0.5930 recall at half the speed.
3. France-specific check: 15% of test has zero training labels.
