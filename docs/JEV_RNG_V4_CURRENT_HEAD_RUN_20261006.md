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
parity is allowed to run. It supports both the full-H8 worker manifest and
the paired standalone current-head builder manifest: the latter explicitly
binds the records sibling and source trace, while partition/order-index fields
are marked not applicable because that producer has no separate files.

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

An authorized supervisor is now armed at
`reproduction_tools/run_authorized_full_h8_pipeline_waiter.sh` (PID recorded
in the runtime log). It waits on the formal report and, only after the strict
authorization helper accepts it, uses the locked 24-video partition at
`/home/liuyeqiang/WWW_jev_full_h8_runtime/partition`. The separate
`small_h8_partition_fixed` three-video fixture is never used as the production
Full H=8 partition.

## Live continuation update — 23:02 UTC

The current branch is
`jev/counterfactual-rng-isolation-v4-20261006`, descending from the pushed
official-runner commit `16b5bc3` and the requested `13a0c679` source gate. The
strict
same-code formal single/chunk equivalence report is `PASS`; the authorized
supervisor has consequently launched the production 24-video Full H=8
scheduler using the locked partition with `1,112,173` main decisions.

At this update the queue is `RUNNING` with `21 PENDING` and `3 RUNNING`
videos. The active workers are video15/video23/video24 on the designated
GPU4/GPU8/GPU9 slots, with live heartbeats. Their outputs are intentionally
kept in worker temporary JSONL files until each video finishes; a missing
per-video final manifest is not treated as completion. The corrected video1,
video6, and video7 current-head builders remain separate and live; none was
stopped, migrated, or replaced.

The current measured Full H=8 worker throughput is approximately
`0.8–1.0 records/second/worker` over a short observation window. With the
currently available three slots this implies roughly `4–6 days` for the full
record build; this is an estimate, not a completion claim, and may change if a
designated slot becomes available. The v6/v7 current-head training waiter is
independent and starts the one-seed Threshold/MLP/JEV screening bundle as soon
as its own corrected builder and parity gates pass; it does not wait for
video1.

The formal official TEST runner is now recorded in
`reproduction_tools/run_full_h8_official_tracking.py`, with its fail-closed
waiter in `reproduction_tools/run_full_h8_official_tracking_waiter.sh`. The
waiter is live and currently waiting for the corrected video1 provenance,
feature parity, candidate parity, stability, closed-loop gate, Full H=8
postprocess, and first-round training reports. Once all are `PASS`, it will
run the three learned controllers in parallel on GPUs 4/8/9 using the same
Stage2 checkpoint, frozen TEST perception cache, and explicit trajectory RNG
seed. It reuses the frozen GMT OFF baseline and never reruns the full OFF
baseline. No official TEST controller result exists yet.

## Live throughput check — 23:05 UTC

A read-only 25-second sample of the three active Full H=8 temporary shards
added `16`, `15`, and `22` JSONL records respectively (`53` total, about
`2.12 records/second` across the three workers). Their worker CPU counters and
temporary-file sizes also increased, so the build is progressing rather than
stalled. At this observed three-slot rate, the remaining 24-video build is
roughly a `5–7 day` wall-clock operation; this remains an estimate because
workloads differ by video and GPU6 may become available later.

The scheduler is polling GPU6 and deferring it only while the live corrected
video6 builder owns PID `33861`; GPU1 is deferred because of the unrelated
external PID `16536`. GPUs 2/3/5/7 remain protocol-reserved and are not used
just because their utilization is low. No process was stopped or migrated.
