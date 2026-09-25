# 03 — Red-team my Amazon ML Challenge plan

## My prompt (summary)

I asked Kiro to act as an adversarial technical reviewer for the Amazon ML Challenge 2026
(a large-scale business entity resolution problem). I provided:

- (A) The full problem statement and rules
- (B) The dataset inventory with exact row counts
- (C) All my empirical findings from exploratory analysis
- (D) My environment and constraints
- (E) The complete plan I intend to execute
- (F) The weaknesses I already suspect (W1–W12)

The task: RED-TEAM the plan. Find logical errors, inconsistencies, arithmetic mistakes,
unstated assumptions, leakage paths, circular reasoning, and silent failure modes.
Output a defect report only — no code, no restatement, no praise.

## Kiro's response (summary)

Produced a full defect report:

- **Arithmetic verification:** confirmed all headline math holds (F_0.5 algebra, oracle
  form, 0.0558 empty-prediction floor, 1.230x shift, 18.7% removal, 39.85% distractor
  rate, 49.8M reverse edges, 25.6GB feature matrix). The danger is not the arithmetic —
  it is what the arithmetic is conditioned on.
- **23 defects (D1–D23)** with severity and a FIX / DROP / MEASURE FIRST verdict each.
- Key critical issues:
  - **D1 / Part 8:** Reverse-primary retrieval has no per-S1 coverage guarantee and fails
    hardest on the low-match entities the macro metric weights most. This is the single
    decision most likely to sink the plan. Replace with a forward-primary generator that
    has a guaranteed per-S1 candidate floor; use reverse retrieval as a feature.
  - **D2 / D3:** The 1.230x shift is attached to the wrong quantity (source targets per
    S1, not true matches per S1). Do not ship shift calibration until an experiment tells
    you which mechanism (more distractors vs more matches vs fewer owners) is real.
  - **D4 / D6 / D10 / D13:** The decision layer is three uncoordinated knobs; replace with
    a single per-entity expected-F_0.5 acceptance objective.
  - **D5:** Two-pass ownership leaks unless assignments are generated out-of-fold.
- Also covered: additional leakage paths, training-label-generator assumptions that may
  not transfer to test, a repair-priority ranking, operational risks (validator OOM,
  memory budget, submission cap), and under-specified steps that could silently lose score.

Full report is in the chat history for prompt 03.
