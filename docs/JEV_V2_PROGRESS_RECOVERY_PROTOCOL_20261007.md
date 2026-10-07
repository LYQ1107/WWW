# JEV v2 progress and recovery protocol

Status: designed and implemented for future frozen-worktree runs. It is not
retrofit into the live video01 v2 process (PID 12163), which remains
untouched.

## Required live progress

Publish an atomically replaced `progress.json` after every completed main
event and before/after each checkpoint. It must include:

```yaml
frame: <int>
view: <int>
event_order: <int>
completed_events: <int>
total_events: <int>
branch_count: <int>
elapsed_seconds: <float>
last_update_utc: <ISO-8601>
schema_version: jev_v2_progress_v1
```

The record JSONL is append-only and fsync'ed after each record. A checkpoint
is atomically replaced only after the corresponding records and progress file
are durable.

## Checkpoint contents

`reproduction_tools/jev_v2_progress_recovery.py` saves:

- the mutable `MutableGMTState`, including active/stale IDs, memory-bank
  contents, association history, counters, and `next_id`;
- explicit trajectory RNG seed, Python RNG state, and call count;
- source commit plus trace/cache/annotation/checkpoint/config and transformer,
  engine, and adapter hashes supplied by the caller;
- the progress record and a state signature.

Resume fails closed if any provenance field differs or the state signature does
not validate. A partial final JSON line is ignored only in a temporary input
snapshot; a checkpoint/records pair is never silently repaired.

## Chunking rule

For a future 24-video canonical build, a chunk boundary must be created by a
deterministic OFF warm-up from the video seed to the boundary. The warm-up
must reconstruct both mutable GMT state and trajectory RNG state, then the
chunk worker may emit only its disjoint decision range. The first controlled
test is single-worker versus chunked on one video with byte/semantic record
comparison before any multi-GPU launch.

The live video01 v2 builder does not expose this protocol and therefore must
not be claimed resumable. It is intentionally left running unchanged.
