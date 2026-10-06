# Corrected video1 reactivation branch fix — 2026-10-06

The corrected video1 builder reached the native reactivation at `(video=1,
frame=214, view=1)` and then failed at `(1, 216, 1)` with an empty mutable
stale bank. A state-only replay of the native OFF trajectory kept stale ID 3
through frame 220, so the failure was not an OFF bank-provenance mismatch.

The failure came from a counterfactual branch that chose `REACTIVATE_OLD` at
frame 214. That branch correctly removed ID 3 from its stale bank, but the
future rollout still applied the OFF trajectory's later
`REACTIVATION_DECISION` event. The builder now keeps strict native/replay
validation for the main OFF path and treats a stale-bank event that is no
longer applicable to a changed branch as branch-local evidence to skip. Typed
MATCH actions are rebuilt separately so an inapplicable reactivation action
cannot overwrite them.

Validation completed before restarting the full rebuild:

- Python compilation: PASS.
- Reactivation semantics regression: PASS.
- Existing regression suite: `3 passed` (one Pillow deprecation warning).
- Same-code OFF state-only diagnostic through frame 220: PASS.

This is a pipeline-fix report only. The full corrected video1 records,
three-question parity, corrected-v4 retraining, and closed-loop tracking remain
pending and must not be inferred from this report.
