# Running the pipeline on Camber Cloud — step by step

Your laptop (15 GB RAM) is too small for the full 26M-row dataset. Camber's SMALL node
(16 CPU / 64 GB RAM) is the target. This guide runs the whole pipeline with one command.

## What gets uploaded

```
Amazon/
  pipeline.py          <- orchestrator
  run_camber.py        <- one-command entry point (install + train + test + validate)
  requirements.txt
  src/                 <- all pipeline modules + validate_submission.py
  data/train/*.tsv     <- 4 train files (200MB + 467MB + 480MB + 121MB)
  data/test/*.tsv      <- 3 test files (167MB + 486MB + 483MB)
```

Do NOT upload: `data/sample/`, `artifacts/`, `output/`, `.git/`, `__pycache__/`.

---

## Step 1 — Open the Camber JupyterHub terminal

In the Camber web app (app.cambercloud.com), open your JupyterHub, then open a Terminal.

## Step 2 — Install the Camber CLI (in the terminal)

```bash
curl -sL https://cli.cambercloud.com/install-v2.sh | bash
camber version
```

## Step 3 — Upload code + data to Stash

From your LOCAL machine terminal (where this repo lives), upload the needed folders.
Replace <PROJECT> with your Camber stash project path.

```bash
# code + config
camber stash cp pipeline.py    stash://majen/<PROJECT>/ --api-key $CAMBER_TOKEN
camber stash cp run_camber.py  stash://majen/<PROJECT>/ --api-key $CAMBER_TOKEN
camber stash cp requirements.txt stash://majen/<PROJECT>/ --api-key $CAMBER_TOKEN
camber stash cp -r src         stash://majen/<PROJECT>/src/  --api-key $CAMBER_TOKEN

# data (large - this is the slow part)
camber stash cp -r data/train  stash://majen/<PROJECT>/data/train/ --api-key $CAMBER_TOKEN
camber stash cp -r data/test   stash://majen/<PROJECT>/data/test/  --api-key $CAMBER_TOKEN
```

## Step 4 — Submit the job (from JupyterHub, in a Python cell or terminal)

```python
import camber.base
job = camber.base.create_job(
    command="python run_camber.py",
    node_size="SMALL",     # 16 CPU, 64 GB RAM
)
print(job.job_id, job.status)
```

`run_camber.py` installs deps, trains, runs test, and validates - all in one go.

## Step 5 — Watch the job

Check status until it finishes. The run prints:
- blocking recall / candidate counts,
- LightGBM training progress,
- in-sample macro F_0.5 (optimistic - real score is lower),
- validator PASS/FAIL.

## Step 6 — Download the results

```bash
camber stash cp -r stash://majen/<PROJECT>/output/ ./output/ --api-key $CAMBER_TOKEN
```

You now have `output/matching_results.tsv` (the scored file) and
`output/candidate_pairs.tsv`.

## Step 7 — Submit to the leaderboard

Upload `output/matching_results.tsv` (and include `candidate_pairs.tsv` in your final
zip). Remember: max 5 submissions/day, 15 total.

---

## Honest expectations

- The pipeline is tested and produces a valid submission. The **real** macro F_0.5 will
  be well below the optimistic in-sample 0.99 you saw locally - your own analysis puts
  the ceiling near ~0.90. Treat the first Camber run as your true baseline.
- If a stage runs slow or hits the node's disk limit, reduce `--top-k` (e.g. 30) to
  shrink candidate volume, or process test in shards.
- Everything is offline: no external data is fetched, satisfying the competition rule.
```
