# Same-score / different-state analysis

## Purpose

This analysis tests the central claim behind JEV: a legacy GMT score or
threshold is not necessarily a sufficient statistic for the action.  It is an
offline diagnostic over the frozen-evidence counterfactual dataset; it is not
an online tracking result and it does not use this analysis to alter test
predictions.

## Fixed protocol

- The dataset is produced by `build_jev_counterfactual_dataset.py` from one
  frozen Stage2 GMT trace and the corresponding annotations.
- For MATCH, MEMORY, and REACTIVATION, the legacy score component is grouped
  within `epsilon=0.01`.
- A pair is retained only when its state-vector L2 distance is at least
  `0.25` and the long-horizon oracle best-action sets are disjoint.
- Every policy is evaluated on exactly the same records, typed question, and
  legal-action mask.  A policy emits an action distribution; it never emits a
  track ID.
- The oracle labels use future ground truth only in the offline
  counterfactual builder.  The report is explicitly marked
  `uses_future_gt=true`; this must not be interpreted as online evidence.

## Reproducible command

The post-Stage2 supervisor writes the real-data report to:

```text
outputs/research_pipeline/decision_stress.json
```

It invokes:

```text
reproduction_tools/analyze_jev_decisions.py
```

The report records pair counts and mean state distance by typed question,
oracle-action disagreement, each policy's action disagreement on the same
pairs, policy checkpoint hashes, parameter counts, NLL/Brier/ECE, and
risk-coverage summaries.

## Interpretation rule

The analysis supports a JEV decision-layer claim only if real pairs exist and
the disagreement is not an artifact of an empty legal-action set, a changed
proposal stream, or future-GT access by the policy.  A zero-pair result is a
negative/insufficient result, not evidence that JEV improves tracking.  Final
numeric values and the GO/NO-GO decision are filled only after the frozen
Stage2 pipeline and its manifest gates pass.

## Current status

The code path and synthetic invariant are passing.  The real report remains
pending the final Stage2 checkpoint and is intentionally not pre-filled with
tracking or policy claims.
