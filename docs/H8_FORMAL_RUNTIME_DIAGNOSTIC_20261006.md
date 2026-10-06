# Formal H=8 runtime diagnostic — 2026-10-06

This report records the first formal-builder failure after the early closed-loop pilot. It is a runtime diagnosis, not a replacement for the frozen GMT baseline or a paper result.

## Snapshot

At the diagnostic snapshot, the formal queue contained 22 `RUNNING` videos, one `FAILED` video, and one `COMPLETE` video. `video_08` was complete with 3,950 records. No running builder was stopped, restarted, or migrated for this diagnosis.

The failed item was `video_04` on `gpu8-slot5`. Its partition contains 7,203 expected decisions; the worker persisted a 4,937-record temporary prefix (18,058,312 bytes) before failing. The prefix is not an official shard because the worker did not reach the complete boundary or produce a manifest.

## Cause

The error was:

```text
RuntimeError: Sizes of tensors must match except in dimension 0.
Expected size 1152 but got size 0 for tensor number 80 in the list.
```

The cache audit found 57,508 indexed payloads in total. Only 29 have zero detections; all 29 are `video_04/view_0` records with the valid representation `pred_boxes=[0,4]` and `reid_features=[0,0]`. There are no records with detections but a zero ReID dimension. Therefore the cache is not corrupted. The failure was an adapter boundary bug: an empty frame inside the 80-step association history was concatenated directly with normal 1,152-dimensional ReID features.

The partial-output audit found zero invalid JSON/schema records in the 4,937 lines. Their event orders are 107,080–112,016 and exactly match the first 4,937 events of the source trace; event 112,017 is the next missing event. Because the prefix does not reach the expected 7,203-record boundary, it is retained only as evidence and is not promoted or merged.

## Fix and validation

`reproduction_tools/jev_gmt_association_adapter.py` now keeps zero-detection rows empty while assigning them the neighboring non-empty feature width only for the transformer concatenation. It does not modify the cache or invent detections. Inconsistent non-empty widths still fail explicitly.

The regression test in `reproduction_tools/test_jev_gmt_association_adapter.py` covers both an empty historical frame and an empty current frame. Validation passed:

- Python compilation;
- GMT association-adapter invariants with empty frames;
- JEV decision invariants;
- JEV policy replay invariants.

## Remaining action

The fix has not been used to claim `video_04` complete yet. The failed shard must be retried with the patched source when a scheduler slot is available. All currently running builders must remain untouched. The partial temporary file is not an official artifact and must not be fed into formal aftercare as if it were complete.

The early pilot conclusion remains unchanged: corrected canonical-feature parity produced `PILOT_GO_FOR_FULL_H8_CONTINUATION`; the pilot is still screening-only and does not replace the full H=8 build, multi-seed work, or official tracking evaluation.
