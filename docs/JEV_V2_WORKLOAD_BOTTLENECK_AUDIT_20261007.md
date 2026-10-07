# Corrected video01 v2 workload bottleneck audit — 2026-10-07

This is a diagnostic report for the live formal builder. It is not a v2
completion report and does not authorize Full H8.

## What the live evidence shows

At `2026-10-07T11:06:42Z`, PID `12163` had been running since
`2026-10-07T04:15:18Z` for `06:51:24`, with `06:51:29` CPU time, `Rsl` state,
100% CPU, and 3,169,444 KiB RSS. Repeated polls showed CPU time increasing.
One Python thread was hot while the remaining threads were sleeping. The
process therefore shows active CPU-bound computation rather than an I/O wait or
an OS-level deadlock. This does not prove that any individual record has
finished because the current run does not expose an incremental record counter.

## Why this run is expensive

The video01 trace has 8,995 typed decision events:

| Question | Events | Legal actions |
|---|---:|---:|
| `MATCH_DECISION` | 4,524 | 3 |
| `MEMORY_DECISION` | 4,297 | 2 |
| `REACTIVATION_DECISION` | 174 | 2 |

The cache contains 1,864 production keys over frames 0–931. For each legal
candidate, the v2 builder clones the pre-decision state, performs the current
step, and replays every cached key through frame + 8. The exact workload
derived from the trace and cache index is approximately:

- 22,514 candidate branches;
- 393,288 total association `step` calls including current-key steps;
- 370,764 future proposal calls;
- approximately 372,637 reactivation-context preparations.

The current builder intentionally buffers records and writes the JSONL and
manifest only after the full build returns, so an empty output directory while
the process is active is expected.

## Operational decision

The live builder is kept on the frozen source commit and is not stopped,
migrated, or restarted. The separate aftercare process continues to wait for
the v2 manifest. Once it appears, provenance, runtime feature parity,
reactivation candidate parity, numerical stability, and the corrected
GMT/Threshold/MLP/JEV closed loop must run in order. Full H8 remains paused and
noncanonical until every gate passes.

The machine-readable details are in
`reports/JEV_RNG_V4/VIDEO01_V2_WORKLOAD_BOTTLENECK_AUDIT_20261007.json`.
