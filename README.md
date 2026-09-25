# Amazon ML Challenge 2026 — Business Entity Resolution

My workspace for the Amazon ML Challenge 2026: given noisy business records from 3
independent sources, find which records refer to the same real-world business.

## Folder structure

```
Amazon/
├── README.md            ← you are here
├── docs/                ← problem statement, guidelines, submission template
├── prompt/              ← log of every prompt I gave the AI + responses
├── data/                ← BIG files (git-ignored, never pushed)
│   ├── train/           ← 7 TSV files, ~2.5 GB, ~26 million rows
│   ├── test/            ← test sources (no labels)
│   └── sample/          ← ~10 MB slice for fast experiments and AI prompts
├── src/                 ← all pipeline code
│   ├── make_sample.py   ← builds data/sample/ from the full data
│   ├── explore.py       ← streaming stats over the full data (chunked, low RAM)
│   └── validate_submission.py  ← run BEFORE every upload
├── output/              ← matching_results.tsv + candidate_pairs.tsv (git-ignored)
└── notes/               ← EDA findings and experiment log
```

## Golden rule for working with this repo

The dataset is **2.5 GB / 26 million rows**. No AI chat can read it, and no script
should load it whole into memory.

1. **Never** paste dataset contents into a prompt.
2. Prototype on `data/sample/` (fits in a prompt, runs in seconds).
3. Ask the AI to **write code**; run the code yourself; paste back only the small summary.
4. On the full data, always read in chunks (`chunksize=`) or with polars/duckdb.

## Useful commands

```bash
# build the small sample (one time, then after any data change)
python src/make_sample.py

# streaming statistics on the full data (safe: reads in chunks)
python src/explore.py

# validate your submission files before uploading
python src/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir data/test
```

## Task reminders

- All files are **tab-separated** (`.tsv`); `sep="\t"` is mandatory in pandas.
- Score is **macro F_0.5** per Source-1 entity (precision weighted 2x over recall).
- Test contains **France**, which is absent from training: treat `country` as an
  open string set, never hard-code `{US, India}`. Every test S1 must appear in the
  submission.
- **No external data lookups** (geocoding, business registries, APIs) — instant
  disqualification.
