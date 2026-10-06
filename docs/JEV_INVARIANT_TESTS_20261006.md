# JEV invariant regression tests — 2026-10-06 08:16 UTC

The current branch was checked without touching the live formal H=8 workers
or any GPU. The GMT environment does not provide the `pytest` module, so each
repository test file was executed through its built-in assertion entry point
with the project `PYTHONPATH`.

All seven test groups passed:

1. Formal H=8 finalizer and explicit source-compatibility invariants.
2. JEV data-contract invariants.
3. Cached-perception counterfactual v2 invariants.
4. JEV decision-model invariants.
5. Online JEV replay and offline-oracle replay invariants.
6. Validation-only selection-gate invariants.
7. GMT/JEV integration invariants.

The offline-oracle replay test explicitly records `future_gt_access=true` only
for the offline oracle path; the ordinary JEV replay test records
`future_gt_access=false`. This confirms that the online replay contract and
the offline label-generation contract remain separated.

Machine-readable details are in
`reports/JEV_INVARIANT_TESTS_20261006.json`.
