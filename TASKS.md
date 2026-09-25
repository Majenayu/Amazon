# TASKS - my (Cline's) track

_Updated 25 Sep 2026, 23:50. Kiro works separately and I never touch its files._

## 1. DONE - the submission is built and validated

| | |
|---|---|
| **`output/matching_results.tsv`** | **1,732,544 rows - every test entity, format PASSED** |
| `output/candidate_pairs.tsv` | 1,732,544 rows, 1,587,779 with candidates |
| `models/v3.joblib` + `v3_meta.json` | trained model, threshold 0.8170 |
| `docs/FINAL_REPORT.md` | full technical report, measured numbers |
| Official validator | **PASS - no blocking issues, safe to submit** |

### The prediction, in numbers

| | |
|---|---:|
| Test candidate pairs generated | 53,405,476 |
| Pairs above the tuned threshold | 4,783,978 |
| Entities with at least one match | 1,423,133 (82.2%) |
| Entities predicted as singletons | 309,411 (17.8%) |

## 2. Headline score

**Validation macro F0.5 = 0.4565** on 441,611 held-out training entities at full
candidate density, against a predict-all-empty floor of **0.0559**. That is
**8.2x better than doing nothing**, and it is measured, not estimated.

| | Value | Source |
|---|---|---|
| Predict-all-empty floor | 0.0559 | `models/v3_meta.json` |
| **Validation macro F0.5** | **0.4565** | `models/v3_meta.json` |
| Best global threshold | 0.8170 | tuned on the exact metric |
| Blocking recall ceiling | 0.5708 | `logs/sweep.log` |
| Implied best possible score | ~0.885 | `1.25r/(0.25+r)` at r=0.5708 |

The leaderboard score will differ from 0.4565 because test blocking ran on the
full 9.97M rows while validation used 30%, and because France has no training
labels. Treat 0.4565 as the best current estimate, not a promise.

## 3. Pipeline - all 5 stages complete

| Stage | Result | Time |
|---|---|---|
| 1. Block train | 44,928,982 pairs, 4.37 GB, 2,400,928 true matches | 72 min |
| 2. Train model | 5.32M rows, threshold 0.8170, **F0.5 0.4565** | 6.8 min |
| 3. Block test | 53,405,476 pairs, 4.85 GB | 93 min |
| 4. Predict | both output files written | 10 min |
| 5. Validate | **PASS** | 8 min |

## 4. Bugs found and fixed during the run

1. **False stage failure.** `full_v3.ps1` printed `STAGE 1/5 FAIL exit=` right after stage 1 had *successfully* written its 4.37 GB output, aborting the run. Cause: PowerShell returns `$null` for `Process.ExitCode` unless the native handle is cached first. Fixed via `$p.Handle`.
2. **No resume.** Added `StageCached`, which skips a stage whose output already exists. Saved 72 min on the relaunch, and 93 min on the second.
3. **`decide.py` missing numpy import** - crashed stage 4 with `NameError`. Fixed. The already-written output files were unaffected.
4. **`predict.py` logging bug** - `{:.0f}` with a `%` tuple raised `TypeError` on the final log line, *after* both files were fully written. Cosmetic, but it made the orchestrator report a false failure. Fixed to `%.0f`.

## 5. Measured negative result, kept not hidden

The metric rewards singletons for predicting nothing *and* rewards recall for entities with matches, which suggested a per-entity rule would beat one global threshold. Built, then measured on 60,000 held-out entities:

| Rule | macro F0.5 |
|---|---|
| best global threshold (t=0.8571) | **0.4562** |
| adaptive, best of 5 m_prior values | 0.4498 |

It loses at every setting, so stage 4 ships the global threshold. Stage 2 already tunes that threshold against the exact metric, which implicitly prices in the singleton payoff. Code kept in `decision.py` / `decide.py`.

## 6. On the 0.9999 target - the honest answer is no

The recall/score relation says 0.9999 needs roughly 99.5% blocking recall *and* perfect classification. We measure 57.08%. That gap is not a tuning problem: about 43% of true matches share **no** token, no 4-char prefix and no postal code with their Source-1 row, so no key-based scheme in this repo can reach them. I will not claim a number I have not measured.

## 7. Next improvements, ranked by expected gain

1. **Raise blocking recall above 0.57** - a second pass over only the Source-1 entities that produced few candidates, using character n-gram keys. This moves the ceiling directly and is worth more than any model change. Estimated ceiling at recall 0.70 is about 0.93.
2. **Wider budget** - post-budget 800 measured 0.5930 recall at half the speed.
3. **France-specific check** - 15% of test with zero training labels.

## 8. Rules kept

- No external data, geocoding, registries or APIs
- `country` treated as an open string set - France never filtered
- All 1,732,544 test entities present in the submission
- Nothing over ~500 MB ever loaded whole into RAM
