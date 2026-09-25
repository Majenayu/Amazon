# WORKFLOW - v2, after the 0.56 leaderboard result

_Written 26 Sep 2026, 00:05. Supersedes the run order in `TASKS.md`._

## What 0.56 actually tells us

Run 1 predicted with a model tuned at **30% density** and applied at **100%
density**. It scored 0.56, above the 0.4565 validation estimate, so the extra
candidates did carry more true matches. But the validation number was measured
under a condition that does not exist at test time, so it was never a fair
predictor of the leaderboard.

Two separate things are now measurable, and they point at different fixes:

| Lever | Run 1 value | Effect |
|---|---|---|
| Blocking recall | ~0.57 at 30% density | caps the score at ~0.885 |
| Threshold calibration | tuned at 20.4 candidates/entity, applied at 30.8 | mis-set, costing real points |

**Fixing calibration is cheap and immediate. Fixing recall is expensive but
larger.** They are independent, so they are done in that order.

## The rule that caused the miss

Never tune a decision threshold on a candidate distribution that differs from
the one you will serve. The `--limit 3000000` shortcut in run 1 was fine for
generating training data quickly, but it silently made the training problem
easier than the real one, and the tuned threshold inherited that easiness.

Concretely:

| | pairs | S1 entities | candidates per entity |
|---|---:|---:|---:|
| run 1 training | 44,928,982 | 2,206,821 | 20.4 |
| run 1 test | 53,405,476 | 1,732,544 | **30.8** |

Every extra candidate is another chance at a false merge, and a false merge on a
true singleton entity costs a full 1.0. A threshold that is safe at 20.4
candidates is too permissive at 30.8.

## New workflow - four stages, each verified before the next

### Stage A - density-matched retrain (`tools/v4_density.ps1`, running now)

1. Re-block training data with **no `--limit`**, so training density equals test
   density (~30.8 candidates/entity).
2. Retrain and re-tune the threshold on that full-density validation set.
3. Re-predict with the corrected threshold. **Test pairs are reused**, not
   rebuilt - only the model changes.
4. Validate the format.

Cost: ~2h15 of blocking plus ~20 min. Expected: the threshold rises above
0.8170, because the optimum is stricter when there is more noise to reject.

**Decision rule: submit v4 only if its validation score at matched density beats
run 1's 0.4565.** Do not ship a worse model because it is newer.

### Stage B - recall, the bigger prize

Blocking recall of 0.57 is the real ceiling. About 43% of true matches share no
token, no 4-char prefix and no postal code with their Source-1 row.

The approach: **a second pass restricted to Source-1 entities that produced few
or no candidates.** Those are the entities the first pass failed on, so a
slower, more expensive method can be aimed only at them. Candidate keys:

- character 4-gram and 5-gram keys over the concatenated name (survives
  transpositions that token keys miss)
- sorted-token-pair keys (survives word reordering and partial names)
- address-street-number + token keys, for records where the name is hopeless
  but the address is clean

Measure recall the same way as before (`probe_recall.py` + `analyze_score.py`)
before committing to a full run. The rule is: **if the probe does not show a
clear recall gain, do not run it on the full data.**

### Stage C - features and model, only after A and B

The 20 features are all cheap string similarities. Worth adding once blocking is
not the binding constraint:

- token-level edit distance on the best-matching token pair
- phonetic codes (Soundex / Metaphone) as a France-safe abstraction
- address normalisation beyond whitespace splitting (street/road/saint aliases)
- a second model in the ensemble, since ranking quality directly sets how many
  candidates survive the threshold

### Stage D - threshold and decision, last

Only worth revisiting after A and B. Run 1 already showed the global threshold
beats a per-entity adaptive rule (`_archive/decision.py`), so do not re-try that
without a materially better probability model.

## Discipline for the rest of the competition

1. **Validate at served density.** Any number quoted as a prediction must come
   from a candidate set built the same way the test set is built.
2. **Measure recall before running.** Every full blocking run costs 1-2 hours.
   The probe costs minutes and has caught every wrong turn so far.
3. **One variable per run.** v4 changes density only. Stage B changes keys only.
   Otherwise an improvement cannot be attributed.
4. **Keep the submission valid at all times.** `output/matching_results.tsv`
   stays the best *verified* result. New runs write to `_v4` files and only
   replace the main file after passing the validator *and* beating the
   incumbent on validation.
5. **Record negative results.** `_archive/README.md` exists so the adaptive-rule
   dead end is not re-explored.

## Time budget

| Stage | Cost | Expected gain |
|---|---|---|
| A - density-matched retrain | ~2.5 h, running | calibration, likely +0.02 to +0.06 |
| B - second-pass recall | ~4 h + 1 h probing | recall 0.57 to 0.65+, ceiling to ~0.91 |
| C - features | ~2 h | small, only after B |
| D - decision rule | ~1 h | small, only after C |

Stage A is the right next move because it is already running, it is cheap, and
it removes a known, quantified defect rather than chasing an unknown one.
