# STATUS

_Updated 26 Sep 2026, 00:16._

## Headline

| | |
|---|---|
| **Leaderboard score** | **0.56** |
| Predict-all-empty floor | 0.0559 |
| Blocking recall (measured) | 0.5708 -> score ceiling ~0.885 |
| Submission | validated PASS, including the ID-existence check |

## Right now

| Stage | State |
|---|---|
| 1. Block TRAIN, full density | **RUNNING** since 00:11 (2.21M S1 x 10.32M rows) |
| 2. Train + re-tune at matched density | queued |
| 3. Block TEST | skipped - `test_pairs.tsv` already full density |
| 4. Predict | queued |
| 5. Validate | queued |

`Get-Content logs\run.log -Wait` to follow. Expect stage 1 to take about 2h05.

## The one change that matters

Training candidates were built from 30% of Source-2/3 rows (20.4 candidates per
entity) and served at 100% (30.8 per entity). The threshold was therefore
calibrated against an easier distribution than the one it met, which is the most
likely reason the leaderboard reads 0.56. `tools/run.ps1` now blocks both splits
with no `--limit`, and stage 3 is skipped because the existing test pairs are
already full density.

## Workspace

Consolidated from ~50 files to 22. The pipeline is 5 modules
(`core`, `block`, `train`, `predict`, `measure`) behind one entry point
(`tools/run.ps1`). Removed 2.1 GB of scratch cache, 3 stale models, ~80 empty
logs, and an 18-file archive of superseded code.

**The merge was verified:** `block.py` reproduces the old `v3block.py` output
byte-for-byte (md5 `b309853b15c41c3993dddfebc6be581e`).

## Files

| Path | What |
|---|---|
| `output/matching_results.tsv` | the submission, 1,732,544 rows |
| `output/candidate_pairs.tsv` | candidate set |
| `data/cache/test_pairs.tsv` | 53.4M test candidates, full density, reused |
| `data/cache/train_pairs.tsv` | being rebuilt at full density |
| `models/v3.joblib`, `v3_meta.json` | run-1 model, threshold 0.8170 |
| `src/baseline/` | the 5-module pipeline |
| `tools/run.ps1` | the only entry point |
| `docs/FINAL_REPORT.md` | technical report, measured numbers |
| `docs/WORKFLOW.md` | what to do next and why |
| `logs/run.log` | one line per stage |

## Commands

```powershell
# follow progress
Get-Content logs\run.log -Wait

# recall probe before any expensive change (minutes, not hours)
.\tools\run.ps1 -Mode recall

# format check before uploading
python src\validate_submission.py --matching output\matching_results.tsv --candidate output\candidate_pairs.tsv --test-dir data\test
```
