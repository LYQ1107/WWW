# Corrected video01 v2 aftercare result — 2026-10-07

The formal corrected video01 v2 artifact completed successfully, but the
runtime semantics gates rejected it for controller selection.

## Passed

- `8995` records were generated from frozen commit
  `4108f18f5040432f68d55872e81e9f76e9acd08f`.
- The manifest and provenance validator passed: zero invalid lines, zero
  duplicate semantic keys, and exact `MATCH/MEMORY/REACTIVATION` counts of
  `4524/4297/174`.
- The corrected Threshold, Generic MLP, and Full JEV checkpoint wrapper
  preflight passed for seed `20261003`.

## Failed gates

- Runtime feature parity: `FAIL`, maximum absolute error
  `0.006004035472869873` against the frozen `2e-5` tolerance. The mismatch is
  concentrated in score-derived features at reactivation boundaries; OFF
  action sequence itself has zero mismatches.
- Fresh reactivation candidate parity: `FAIL`. Candidate IDs/order and legacy
  OFF actions match, but native/replay scores and chosen proposal IDs do not;
  the capped accounting is `225` mismatches with `100` native-only and `100`
  replay-only entries.
- Three repeated runtime stability replays: `FAIL` identically on all three
  repetitions (`575680` feature errors per repetition).

Therefore the corrected three-way tracking closed loop was not run, and
`FULL_H8_AUTHORIZED=false` remains in force. No tracking metric from this
aftercare run is a paper result.

## Repair path

The leading code-level discrepancy is in the stale-bank transformer input:
the current adapter used zero boxes and separate historical `Instances` for
old IDs, while native GMT retains each old ID's original proposal box and
concatenates the stale bank into one historical `Instances` object. A fix is
being validated in a separate worktree with targeted regression tests. The
formal artifact above is retained unchanged for auditability; it will not be
silently overwritten.

Machine-readable details:
`reports/JEV_RNG_V4/VIDEO01_V2_AFTERCARE_GATE_RESULT_20261007.json`.
