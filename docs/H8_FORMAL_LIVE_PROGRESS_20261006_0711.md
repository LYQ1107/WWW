# Formal H=8 live progress snapshot — 2026-10-06 07:11 UTC

This is a point-in-time operational snapshot. It is not a final dataset
result and does not authorize training, selection, or paper claims.

## Observed state

- Queue observation: `2026-10-06T07:11:42.345435+00:00`.
- Live monitor observation: `2026-10-06T07:11:39.364469+00:00`.
- Formal queue: 22 `RUNNING`, 1 `COMPLETE`, 1 `FAILED`.
- Completed shard: `video_08`, 3,950 records, with its COMPLETE manifest.
- Persisted records: 125,720 / 1,112,173 expected.
- Persisted throughput: 2.4381266429856905 records/second.
- Sliding-window estimate: `2026-10-10T23:34:54.126284+00:00` UTC.
- Active scheduler and 22 worker processes were live at observation time.

## Failed shard and safety decision

`video_04` remains failed after writing a verified 4,937-record prefix out
of 7,203 expected decisions. The prefix is not an official shard. The empty
ReID-frame adapter fix and its regression tests are already pushed, but no
safe formal worker slot is currently available. The queue was not modified,
and no running builder was stopped, restarted, or migrated.

The retry also remains subject to the strict source-provenance gate: the
existing formal shard lineage must not be silently mixed with a patched
source revision. A retry will be launched only after an explicit, auditable
lineage/equivalence decision.

## Reproducibility anchors

- Canonical Stage2 checkpoint SHA256:
  `cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8`.
- Frozen GMT baseline is unchanged and is not rerun.
- This snapshot records operational evidence only; the early closed-loop
  pilot remains screening-only.
