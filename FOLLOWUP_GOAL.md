# GMT Causal State-Recovery Audit Follow-up

This follow-up is an audit of the existing GMT release-code reproduction. It does not retrain GMT and does not delete or rewrite any first-round causal artifacts.

## Objective

Determine, using event-isolated counterfactual replay over a fixed perception stream, whether corruption of GMT identity state causes future association errors and whether quarantining or repairing that state reduces those errors. Only if the replay-fidelity gate, snapshot round-trip gate, and the predefined STRONG STATE-CAUSAL GO gate all pass may a JEV-GMT prototype be designed.

## Frozen constraints

- Preserve every existing manifest, result, log, output, and first-round run under `causal/`.
- Do not retrain GMT.
- Use fixed seed `20260930`, `TEST_LEN=40`, released memory-bank settings, and no test-GT tuning.
- Every C1b/C2b event branches from the same baseline pre-event state; no long rollout with multiple interventions.
- Stop on any replay-fidelity or snapshot-roundtrip failure.
- Stop and do not build JEV if C2b is positive but neither C1b quarantine nor C1b repair reduces future error at the required horizons.

## Deliverables

`FOLLOWUP_STATUS.md`, the association replay cache and engine, fidelity and snapshot gate manifests, frozen C1b/C2b manifests and results, statistical reports, and `causal_followup/STATE_CAUSAL_EVIDENCE_MATRIX.md`. JEV source/design work is conditional on the strong causal gate only.
