# Checkpoint and frozen-baseline audit — 2026-10-06

This audit preserves the completed upstream evidence while formal H=8 construction continues.

## Checkpoints

- Stage1 `model_16000.pth`: SHA256 `143e84deb50bdf5379c8f4463f1b9b237132e9281726b9cff469aff8c9dbe64e`; checkpoint validation and reload: `PASS`.
- Stage2 `model_20000.pth`: SHA256 `cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8`; checkpoint validation and reload: `PASS`.

## Frozen GMT baseline

The canonical Stage2 baseline remains frozen and was not rerun. The result is tied to the Stage2 SHA above:

```text
HOTA 67.442   DetA 66.278   AssA 68.992   IDF1 82.239
MOTA 80.942   IDSW 3092     Frag 8004
CVIDF1 79.0248   CVMA 80.9276
```

The OFF equivalence audit, corrected TrackEval evaluation, and corrected cross-view evaluation all report `PASS`.

## Current boundary

Formal H=8 remains incomplete: 22 shards are running, one (`video_08`) is complete, and `video_04` failed on the pre-fix empty-ReID-frame adapter path. The H=8 dataset is therefore not yet finalized, and no formal three-controller selection or official tracking comparison is claimed by this audit.
