#!/usr/bin/env python3
"""One-command Camber entry point: install deps, then train + test end to end.

Run on a Camber SMALL node (16 CPU / 64 GB). It:
  1. installs requirements,
  2. trains the matcher on data/train (reports in-sample macro F_0.5),
  3. runs test on data/test and writes output/matching_results.tsv +
     output/candidate_pairs.tsv for EVERY test S1 entity (US, India, France),
  4. runs the provided validator so you know the files are submittable.

Expected layout on Camber (upload preserves this):
    <workdir>/
      pipeline.py, run_camber.py, requirements.txt
      src/...            (all pipeline modules + validate_submission.py)
      data/train/*.tsv   (the 4 train files)
      data/test/*.tsv    (the 3 test files)

Usage on Camber:
    python run_camber.py
"""

from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def sh(cmd: list[str]):
    print("\n>>> " + " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=HERE)


def main() -> int:
    # 1. Dependencies.
    sh([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"])

    # 2. Train (builds maps, blocks, features, trains model, reports macro F_0.5).
    sh([sys.executable, "pipeline.py", "train",
        "--data-dir", "data", "--artifacts", "artifacts", "--top-k", "50"])

    # 3. Test (writes output/ for every test entity including France).
    sh([sys.executable, "pipeline.py", "test",
        "--data-dir", "data", "--artifacts", "artifacts",
        "--output", "output", "--top-k", "50"])

    # 4. Validate the submission format (never rejects, just reports).
    validator = os.path.join("src", "validate_submission.py")
    if os.path.isfile(validator):
        sh([sys.executable, validator,
            "--matching", "output/matching_results.tsv",
            "--candidate", "output/candidate_pairs.tsv",
            "--test-dir", "data/test"])

    print("\nAll done. Download the output/ folder from Stash.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
