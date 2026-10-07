# JEV RNG v4 live progress — corrected video01 gate first

**Latest authoritative refresh:** `2026-10-07T12:05:14Z`
**Current decision:** `NO_GO_RUNTIME_SEMANTICS_GATE`

## Live status

| Work item | Current state | Authority |
|---|---|---|
| corrected video01 v2 builder | **COMPLETED**, PID `12163`, 8995 records and manifest PASS | runtime artifact + manifest |
| video01 v2 aftercare | **COMPLETED NO-GO**; parity/candidate/stability failed | `VIDEO01_V2_AFTERCARE_GATE_RESULT_20261007.json` |
| Full H8 original scheduler | **GRACEFULLY PAUSED** | queue state |
| Full H8 slot2/slot3 supervisors and workers | **GRACEFULLY PAUSED** | queue state/process check |
| Full H8 canonical authority | **FALSE** | source-commit audit |
| corrected video06/video07 three-way training | complete; preserve report; no retraining | existing report |
| segmented small gate | **LIVE**, PID `1819`, `8281/17491` records including running partials | MPS switch/equivalence report |

The Full H8 run is not “running in the background” anymore. It is a preserved
speculative build and must not be resumed from the current queue.

The separate segmented small gate is still allowed to run. Its latest queue is
`55 COMPLETE / 28 RUNNING / 5 PENDING` across video01/video06/video07, with
`13734/17491` records counted including running partials. The
MPS transition was checked on two bounded probes: `70/70` early records and
`102/102` late records were exact, with zero late candidate mismatches. This
does not authorize Full H8 or create a final paper result.

## What was stopped and what was not

The following Full H8 process groups received graceful `SIGTERM`; no `kill -9`
was used:

- original scheduler PID `19950`;
- original workers `19972`, `19992`, `20007`, `13063`;
- slot2 workers `40330`, `40345`, `40354`, `40364`;
- slot3 workers `7418`, `7425`, `7439`, `7453`;
- Full H8 official/authorized waiters `35636`, `35638`, `35640`, `36327`,
  `36329`, `36332`.

All selected Full H8 PIDs exited after the graceful pause. Corrected video01
PID `8251` and its aftercare processes were checked separately and remained
alive.

## Preserved Full H8 runtime evidence

Queue:
`/home/liuyeqiang/WWW_jev_rng_v4_runtime/full_h8_current_head/queue_state.json`

The queue now records:

```text
status                    = PAUSED
canonical_authority       = NOT_CANONICAL
build_classification       = PRE_VIDEO01_GATE_SPECULATIVE_BUILD
source_commit_authority    = MIXED_SOURCE_COMMIT_RISK
```

It has `12 PAUSED` entries and `12 PENDING` entries, with zero official
`COMPLETE` shards. The temporary JSONL files contain approximately `21,217`
complete visible lines. They remain untouched for debugging and throughput
analysis. Logs, queue lock, scheduler logs, resource manifest, and PID
snapshot are also retained.

The resource manifest was last written before the pause and therefore remains
a historical scheduling snapshot (`status=RUNNING` at its own timestamp). The
queue pause record and the GitHub pause report supersede that stale live-status
field; the manifest itself was not deleted or treated as canonical evidence.

## Why the previous Full H8 was invalid as canonical data

The build started around `2026-10-06T22:52Z` from mutable worktree
`/data1/liuyeqiang/WWW_rng_fix_v4`. After the original launch, the following
reactivation-semantic changes landed:

- `03f1dfd98f23ae4baab53e9b994bbdcbb73122a6` — branch-local reactivation
  semantics (`23:23Z`);
- `68bba4b33deeaf33b9dfc6f8e99556e4bc8725e7` — video1 reactivation branch fix
  documentation/state (`23:33Z`).

The original, slot2 and slot3 worker groups were not proven to have one common
source commit. The available prelaunch HEAD observations differ (`7594b1d`,
`a942dd8`, `e922603`), and the workers did not record their exact startup SHA.
Therefore the build is permanently classified as mixed-source speculative
output, not as a canonical dataset.

## Completed evidence that remains valid

- Frozen GMT OFF baseline is reused; no full VISION_test OFF rerun is needed.
- Stage1/Stage2 checkpoints and CPU regression audit remain valid.
- Corrected video06/video07 provenance and runtime parity gates remain valid.
- Formal video07 single-worker versus chunked equivalence remains a chunking
  implementation result, not permission to skip video01.
- `CORRECTED_V4_SMALL_H8_THREE_WAY_VIDEO06_VIDEO07.json` remains the existing
  single-seed screening report. Its corrected ECE/Oracle aggregation result is
  preserved; do not retrain those three models merely because Full H8 was
  paused.

## Current gate order

```text
fix stale-bank geometry/runtime semantics and regenerate video01 v2
  ↓
video01 builder COMPLETE + provenance
  ↓
MATCH / MEMORY / REACTIVATION runtime feature parity
  ↓
reactivation candidate parity
  ↓
numerical stability
  ↓
GMT OFF / corrected Threshold / corrected MLP / corrected JEV closed-loop
  ↓
GO / NO-GO
  ↓ only GO
freeze exact commit and rebuild Full H8
```

The consolidated v2 aftercare file
`reports/JEV_RNG_V4/VIDEO01_V2_HARD_GATES_20261007.json` is `PENDING` with
`FULL_H8_AUTHORIZED=false`. The aftercare result is an explicit NO-GO because
runtime feature parity, candidate parity, and stability failed. It must not be
manually changed to PASS.

## Future Full H8 safety contract

On a GO decision, set one exact SHA and create:

```text
/data1/liuyeqiang/WWW_h8_frozen_<SHA>
```

The updated scheduler and worker enforce:

- exact frozen worktree path and `git rev-parse HEAD` equality;
- queue binding to `canonical_h8_commit` and `source_worktree`;
- scheduler recheck before every new worker launch;
- worker manifests containing `source_commit`, `transformer_sha256`,
  `counterfactual_engine_sha256`, and `adapter_sha256`;
- authorization requiring every named corrected-video01 gate in addition to
  formal chunk equivalence.

Until then, the preserved Full H8 `.tmp` records are not training data and the
project has no final JEV tracking claim.
