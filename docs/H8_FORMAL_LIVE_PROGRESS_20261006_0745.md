# Formal H=8 live progress snapshot — 2026-10-06 07:45 UTC

This is a point-in-time operational snapshot. It is not a final dataset
result and does not authorize training, selection, or paper claims.

## Observed state

- Queue observation: `2026-10-06T07:45:10.737851+00:00`.
- Live monitor observation: `2026-10-06T07:44:55.155301+00:00`.
- Formal queue: 20 `RUNNING`, 3 `COMPLETE`, 1 `FAILED`.
- Completed shards: `video_06` (5,164 records), `video_07` (3,337), and
  `video_08` (3,950), each with an audited COMPLETE manifest.
- Persisted records: 130,626 / 1,112,173 expected.
- Persisted throughput: 2.4158970159330884 records/second.
- Sliding-window estimate: `2026-10-11T00:36:22.026833+00:00` UTC.
- The scheduler and all 20 formal workers were live at observation time.

The monitor deliberately excludes stale and current temporary-file lines from
the official persisted-record count; it reported 131,757 such lines. They are
not treated as complete records or merged into the formal dataset.

## Failed shard and safety decision

`video_04` remains failed after writing a verified 4,937-record prefix out of
7,203 expected decisions. The prefix is not an official shard. The empty
ReID-frame adapter fix and its regression tests are already pushed, but the
currently occupied safe GPUs do not provide a safe slot for the required
GPU-backed equivalence check. The queue was not modified, and no running
builder was stopped, restarted, or migrated.

The retry remains subject to the strict source-provenance gate: existing
formal shards must not be silently mixed with a patched source revision. A
retry will be launched only after an explicit, auditable lineage/equivalence
decision.

## Reproducibility anchors

- Canonical Stage2 checkpoint SHA256:
  `cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8`.
- Frozen GMT baseline is unchanged and is not rerun.
- The early closed-loop pilot remains screening-only.
- This snapshot records operational evidence only; it is not a final result.

