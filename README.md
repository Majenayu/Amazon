# Amazon ML Challenge 2026 - Business Entity Resolution

Match noisy business records across three independent sources: given a Source-1
business, find every Source-2 and Source-3 record for the same real-world
business.

## Current result

| | |
|---|---|
| **Leaderboard score** | **0.56** |
| Predict-all-empty floor | 0.0559 |
| Measured blocking recall | 0.5708 |
| Submission | `output/matching_results.tsv`, 1,732,544 rows, validator PASS |

## Run it

```powershell
.\tools\run.ps1                  # full pipeline, train and test at MATCHED density
.\tools\run.ps1 -Mode recall     # blocking-recall probe - run this before anything slow
.\tools\run.ps1 -Mode sweep      # blocking parameter sweep on a small window
.\tools\run.ps1 -Mode predict    # re-predict only, reusing existing candidate pairs
```

Resumable: a stage whose output already exists is skipped, so re-running is safe.

Always validate before uploading:

```powershell
python src\validate_submission.py --matching output\matching_results.tsv --candidate output\candidate_pairs.tsv --test-dir data\test
```

## How it works

```
train_source1 ---> normalise ---> union key index --+
                                                    +--> ranked candidates --> 20 features
train_source2/3 ---> normalise ---> probe keys ----+                                   |
                                                                                     v
                                              HistGradientBoosting --> probability --> threshold
                                                                                     |
                                                    output/matching_results.tsv <----+
```

Test has 1,732,544 x 9,969,589 = about 1.7e13 possible pairs, so blocking is
mandatory, and **blocking recall is the hard ceiling on the score**. At recall
0.5708 the ceiling is about 0.885 even with a perfect classifier.

## The one rule that matters most

**Never tune a decision threshold on a candidate distribution that differs from
the one you will serve.** Run 1 blocked training data from 30% of the Source-2/3
rows (20.4 candidates per entity) and served the model on 100% (30.8 per
entity), so its threshold was calibrated against an easier problem than the one
it met. The leaderboard returned 0.56. `tools/run.ps1` now blocks both splits
with no `--limit`. Details in `docs/WORKFLOW.md`.

## Files

```
|-- README.md               this file
|-- STATUS.md               live progress
|-- TASKS.md                task list
|-- docs/
|   |-- FINAL_REPORT.md     technical report, all measured numbers
|   |-- WORKFLOW.md         what to do next, and why
|   |-- problem_statement.md / .pdf, guidelines.pdf
|   `-- Documentation_template.md
|-- prompt/                 log of prompts given to the AI
|-- data/                   BIG files, git-ignored, never pushed
|   |-- train/ test/        7 TSVs, 2.35 GB, 26,435,994 rows
|   |-- sample/             ~12 MB labelled slice for fast iteration
|   `-- cache/              generated candidate pairs
|-- src/
|   |-- baseline/           THE PIPELINE - 5 files
|   |   |-- core.py         normalisation, blocking keys, the 20 features
|   |   |-- block.py        candidate generation (one streaming pass)
|   |   |-- train.py        train, tune the threshold, or score a prediction
|   |   |-- predict.py      score, threshold, write the submission files
|   |   `-- measure.py      blocking recall and the recall/volume trade-off
|   |-- validate_submission.py   official format checker
|   |-- explore.py          streaming dataset stats
|   `-- make_sample.py      build data/sample/
|-- tools/run.ps1           the only entry point
|-- models/ output/ logs/   generated, git-ignored
```

`src/*.py` at the top level, `pipeline.py`, `run_camber.py`, `artifacts/` and
`camber_*.zip` belong to a separate parallel effort and are not used here.

## Output format

Tab-separated, two columns, one row for **every** test Source-1 entity
(1,732,544 rows). An empty second column means "no match", which is a valid
scoring answer. A missing row scores zero, so all rows are always emitted.

```
source1_entity_id <TAB> matched_entity_ids
S1-714132312 <TAB> S2-187020300,S2-435263846,S3-625880872
S1-106407869 <TAB> S2-705547832
```

## Rules this solution obeys

- No external data, geocoding, registries or APIs - instant disqualification
- `country` is an open string set; France (~15% of test) has no training labels
  and is handled by language-agnostic features
- No file over ~500 MB is ever loaded whole into RAM
- The official validator passes before every submission

## Why the AI could not read the data directly

The dataset is 2.52 GB / 26.4M rows. An AI context window holds roughly 1 MB of
text, so asking an AI to "read the dataset" is not slow, it is impossible. The
working pattern is: **ask the AI to write code, run it locally, paste back only
the small summary.** `data/sample/` exists for exactly this reason.
