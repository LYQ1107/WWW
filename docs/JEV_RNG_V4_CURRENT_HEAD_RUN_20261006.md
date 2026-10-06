# Corrected-v4 current-HEAD run log — 2026-10-06 UTC

This run log records the fail-closed transition from the earlier artifacts to
the current-HEAD artifacts. It is a progress/provenance document, not a
tracking result.

## Why a second rebuild was started

The existing files under
`/home/liuyeqiang/WWW_jev_rng_v4_runtime/small_h8_rng_controlled_v3_canonical_features`
have `source_commit=ec40eff`. That revision predates the production-seed and
mutable-state corrections introduced by `0c31b72`. A three-repeat replay of
the existing v7 artifact was deterministic but failed runtime parity:

- compared records: `1699/1705`;
- OFF-action mismatches: `10`;
- maximum feature error: `1685.5164794921875`;
- p99 feature error: `25.96832275390625`.

Therefore those artifacts are retained for audit/screening, but their
checkpoints are not eligible for corrected-v4 final tracking.

## Current-head artifacts

The new workers were launched from the pushed branch
`jev/counterfactual-rng-isolation-v4-20261006` at commit `5cd989d` (the
subsequent documentation-only commit is `a8a96e4`) and use the frozen GMT
checkpoint, perception cache, trace partition, H=8, and `seed=20261003` policy
plan:

| Artifact | Runtime path | Status at launch | GPU |
|---|---|---:|---:|
| video6 current-head rebuild | `/home/liuyeqiang/WWW_jev_rng_v4_runtime/small_h8_rng_controlled_v4_current/video_06` | running | 5 |
| video7 current-head rebuild | `/home/liuyeqiang/WWW_jev_rng_v4_runtime/small_h8_rng_controlled_v4_current/video_07` | running | 3 |
| video1 current-head replacement | `/home/liuyeqiang/WWW_jev_rng_v4_runtime/small_h8_rng_controlled_v4_current/video_01` | queued; starts after GPU8 formal single | pending |
| formal video7 single reference | `/home/liuyeqiang/WWW_jev_rng_v4_runtime/formal_same_code_video7_v6_current/single/video_07` | running | 8 |
| formal video7 chunk warm-up | `/home/liuyeqiang/WWW_jev_rng_v4_runtime/formal_same_code_video7_v6_current/warmup` | PASS, 3 snapshots | 7 |
| formal video7 chunk0/chunk1 | `/home/liuyeqiang/WWW_jev_rng_v4_runtime/formal_same_code_video7_v6_current/chunked/video_07` | running | 0/7 |

The in-flight corrected video1 worker and the older video6/formal workers were
not stopped or migrated. Their outputs remain separate and will be checked by
source/provenance gates before use. Because the in-flight video1 process was
started before the current production-seed/state corrections, a current-HEAD
video1 replacement is queued rather than silently treating the older output as
final.

## Native cache replay diagnostic

The canonical frozen perception cache was replayed through native GMT OFF on
video1, using the same 932-frame train sequence and the frozen
`model_20000.pth`. The run completed successfully and produced 8,995 decision
trace lines: MATCH/MEMORY/REACTIVATION counts were `4524/4297/174`.
`pred_classes` is restored from the cache proposal metadata so the native
evaluator output schema remains intact. This is native trace provenance only;
it is not a native-vs-mutable candidate-parity result. The evidence is recorded
in `reports/JEV_RNG_V4/NATIVE_GMT_CACHE_REPLAY_VIDEO01_CORRECTED.json`, while
the final candidate gate still waits for the current-head video1 records.

## Required order after builders finish

1. Validate current-head manifests, SHA, exact question counts, finite state,
   checkpoint/cache/trace provenance, and `H=8`.
2. Compact current-head v6/v7 and create the sequence-disjoint split with
   `seed=20261003`.
3. Train exactly one Threshold, MLP, and JEV controller (20 epochs, matched
   optimizer/LR/batch settings), then calibrate and aggregate into
   `CORRECTED_V4_SMALL_H8_THREE_WAY_VIDEO06_VIDEO07.json`.
4. Finish the current-head formal single/chunk build and require the exact
   3337-key equivalence verifier to PASS before authorizing chunking.
5. Run the required three-repeat corrected video6 parity stability probe at
   candidate tolerance `2e-5`; structural fields remain exact regardless of
   numeric tolerance.
6. Finish video1 provenance, separately report MATCH/MEMORY/REACTIVATION/TOTAL
   parity, rerun candidate parity on the matching perception source, then run
   wrapper parity and only afterward the corrected closed-loop comparison.

The post-builder provenance gate is implemented by
`reproduction_tools/validate_jev_video_artifact.py`. It refuses incomplete or
source-mixed artifacts and checks the manifest-bound record/trace/order/
checkpoint/annotation hashes, JSONL semantic-key uniqueness, finite 64-D
canonical state features, and exact typed-question counts before runtime
parity is allowed to run.

The runtime parity wrappers now take an explicit numeric tolerance instead of
silently mixing the historical `1e-6` and `1e-4` values. The current candidate
is `2e-5`, and the full wrapper records exact expected/runtime counts for each
of MATCH, MEMORY, REACTIVATION, and TOTAL. That candidate is not a final
authorization until all three corrected video6 stability repetitions pass.

Until all gates pass, the verdict remains `BLOCKED` for corrected-v4 final
tracking and for the 24-video canonical H=8 rebuild.

## Live continuation update — 22:30 UTC

The historical launch table above is retained for provenance. The active
artifacts are now the separate current-head paths below; none of the old
small-gate records or checkpoints is used for corrected-v4 final tracking.

- Branch: `jev/counterfactual-rng-isolation-v4-20261006`, pushed HEAD
  `9935053`, a descendant of the requested `13a0c679` gate commit.
- Corrected video1 rebuild: `formal_current_head_corrected_full/` from the
  current-head OFF trace, expected `8995` records; the builder remains live
  and its output is intentionally committed only after the final manifest is
  complete.
- Corrected video6/video7 rebuilds: the same `formal_current_head_corrected_full/`
  root, expected `5162` and `3334` records; both builders remain live.
- Same-code formal video7 gate: the three chunk artifacts are complete and
  the single-worker reference is still running (`2430/3337` records at the
  last check). The verifier will write
  `FORMAL_GMT_INTRA_VIDEO_CHUNK_EQUIVALENCE_VIDEO07_CURRENT_HEAD_V10.json`
  only after the single manifest is complete.
- After video1 completion, the active aftercare order is: provenance, full
  four-way question-count/runtime-feature parity, native-vs-mutable
  reactivation candidate parity, then three-repeat numerical stability. The
  closed-loop waiter consumes only these current-head reports and the new
  current-head v6/v7 training bundle.

The frozen GMT baseline is still reused without rerun, and the 24-video H=8
build remains fail-closed until the same-code formal equivalence report has a
complete `PASS` final gate.
