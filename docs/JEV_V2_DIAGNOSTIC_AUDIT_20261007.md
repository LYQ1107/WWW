# Video01 v2 diagnostic audit — 2026-10-07

This audit was run while the live corrected video01 v2 builder (PID 12163)
continued unchanged. It is diagnostic evidence, not a v2 acceptance or paper
result. The speculative Full H8 build remains paused and noncanonical.

## 1. Recalculation scope

The tracked diff from the old artifact commit `60675dae` to the frozen current
commit `4108f18f` is limited to `reproduction_tools/build_jev_counterfactual_v2.py`
(11 insertions, 4 deletions): it records `model.thred_bank` separately as
`bank_threshold` and uses that value for the reactivation state feature. The
counterfactual engine, adapter, transformer, config, trace, cache, annotation,
checkpoint, and RNG/component hashes were checked.

The bounded regression evidence is decisive for reuse policy:

| Check | Result |
|---|---|
| First 20 event keys/order | exact |
| First 20 state projection/digest | 20/20 exact |
| First 20 action outcome/label projections | 0/20 exact |
| Same 4108f18 diagnostic repeated twice | byte-identical (`25092f9a…`) |
| Independent 60675 fixed-worktree replay | exact match to current diagnostic, not old artifact |

The old 8995-record artifact has SHA-256
`22662cb1b5441de3d69582b49d862ccde3228b678e74c5028004e557e3f11daa`; the
independent 20-record replay and current diagnostic both have
`25092f9afa8d53e160e8a456b95bb75d5f466ddc7644564450978ca733bc803c`.
Therefore the old artifact cannot be reproduced from the recorded inputs and
committed 60675 source alone. Its hidden generation difference is not
recoverable from the manifest.

Decision: do not reuse old `action_outcomes`, `horizon_outcomes`, target
probabilities, best actions, or sample weights. The current v2 artifact must
recompute them all. Exact state equality on 20 records is observation only,
not permission for partial reuse.

## 2. Candidate parity

The native v3 and v4b OFF traces are byte-identical:

`cb340485ecf348d2a63cacf69bcf9d3c3fd78a949a7d1cb8f5a9738d5c5a5e1b`

Both contain 8,995 lines. The previous replay parity report remains blocked:

- 225 capped mismatches;
- 100 native-only event keys and 100 replay-only event keys;
- 25 candidate-score tolerance failures;
- 8 chosen-proposal-ID mismatches;
- candidate ID set/order and legacy OFF action checks pass.

At the first score failures, both bank thresholds are `0.4`, so the mismatch
is not explained by the `bank_threshold` feature fix. The old replay is not a
valid parity witness. Fresh v2 candidate parity must be run against the
completed v2-bound artifact, including the mismatch-covering events, before
controller selection or Full H8 authorization.

## 3. Measured cost bottleneck

Two independent 20-event diagnostics on idle GPU 4 were deterministic. The
second run measured 22.53 seconds wall time:

| Operation | Calls | Time |
|---|---:|---:|
| proposal | 857 | 8.72 s |
| step | 908 | 8.21 s |
| state clone | 51 | 0.019 s |
| reactivation context | 862 | 0.0035 s |
| utility scoring | 51 | 0.036 s |

Proposal plus step consumed about 75% of wall time in that run (about 70% in
the first run). Clone and utility scoring are not the bottleneck. The sample
expanded to 51 branch clones, 908 state steps, and 857 proposals for 20
records; these counts are consistent with the rough branch/step hypotheses,
but are not an ETA estimator for the full video.

## 4. Progress/recovery implementation

`reproduction_tools/jev_v2_progress_recovery.py` and
`docs/JEV_V2_PROGRESS_RECOVERY_PROTOCOL_20261007.md` implement the future
protocol: atomic progress publication, fsync'ed record append, checkpointed
mutable state plus explicit trajectory RNG and provenance, and fail-closed
resume validation. The live PID 12163 was not modified; progress cannot be
retrofit by editing its output directory.

## Current gate

At the audit snapshot PID 12163 was still CPU-active and no v2 records or
manifest had been published yet. Keep it running. After completion, bind all
acceptance checks to its new manifest, then run v2 provenance, runtime feature
parity, candidate parity, numerical stability, and the corrected three-way
closed loop. Only a passing video01 gate can authorize a canonical Full H8
build.
