# VGGT point association experiment record

## Component admission

Official local VGGT-1B weights match the author's Hugging Face LFS pointer:
`d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`.
The clean official source is `a288dd0f14786c93483e45524328726ab7b1b4ce`.
Four actual forward calls (both anchor orders, twice) passed. Aggregator and
TrackHead are strictly loaded; unused camera/depth/point heads are disabled.
Preprocessing is bit-exact to official 518 pad mode. First-frame query error
is 0.000061 original pixels. The two-image/16-point warmed component calls
take about 1.8 seconds per anchor; the measured peak allocated memory is
4,455,105,024 bytes. These are component timings, not MOT E2E FPS.

The first attempt exposed the official memory optimization's unused `None`
token layers. Its error log is preserved. Casting only present layers fixed
the adapter; subsequent forwards used the official pretrained weights.

## First causal system pilot: Scene 8, first 32 TRAIN timestamps

Fixed preregistered defaults: 16 points/person, visibility/confidence >= 0.5,
minimum 3 inside points, weighted support >= 0.20, maximum reference gap 3.
The GMT detector is executed live in its own environment and its candidates
and independent ROI features are verified against the same immutable cache
on every RGB pair. No GT is used by inference.

| Diagnostic | Actual result |
|---|---:|
| Synchronized timestamps | 32 |
| Raw GMT candidates / final boxes | 128 / 128 |
| GT-matched boxes at IoU 0.5 | 128 |
| Predicted Global IDs | 4 |
| Same GT identity detected in both cameras | 64 events |
| Correct shared Global ID | 0 / 64 |
| GT-consistent accepted point-support pairs | 124 / 124 |
| True RGB-to-final-ID E2E FPS | 0.213905 |
| All VGGT forward seconds, 32 timestamps | 126.999309 |

The 124 consistent transfers are local temporal matches. They do not prove
cross-camera identity association: in this pilot the two cameras retain
separate IDs for the same two people. There are no wrong cross-camera merges
in the post-hoc GT check. Median cross-view point confidence on the first
pair is about 0.23 (A->B) and 0.34 (B->A), below the default 0.5 gate; local
temporal confidence is around 0.99. Threshold choices must be assessed on the
complete registered TRAIN validation, including incorrect associations.

The pipeline's first version also only synchronously merges births. Existing
separate IDs require an explicit, audited late-consistency merge mechanism.
This is an implementation limitation to fix before the final A3 comparison.
The pilot uses its recorded continuous edge coordinate convention; the next
coordinate revision will explicitly account for PIL pixel-center resampling.
Both changes require a new pilot/result directory and a recorded source SHA.

No TEST metric, final ranking or independent geometry gain is claimed here.
Full evaluation remains outstanding. A negative pilot does not end either
registered route.

## Artifacts and reproduction

* Pilot: `/data1/liuyeqiang/nas/A100/WorldMOT/vggt_point_association_20261023/train_joint_smoke32_v1/scene_08/`
* Offline diagnostic: `/data1/liuyeqiang/nas/A100/WorldMOT/vggt_point_association_20261023/train_joint_smoke32_diagnostic_v1.json`
* Component artifacts: `results/vggt_point_component_preflight.json`.

From the point worktree, use `PYTHONPATH=worldmot` with the existing
`/data1/liuyeqiang/envs/worldmot311/bin/python`. Run
`worldmot/scripts/run_vggt_point_fullvideo.py --split train --metadata
/data1/liuyeqiang/nas/A100/WorldMOT/fulltrain_20261022/audit/train_metadata.json
--raw-cache /data1/liuyeqiang/nas/A100/WorldMOT/gmt_detector_comparison_20261023/raw_train_v1
--scenes 8 --limit 32 --method joint --detector-mode live_verified --output NEW_DIRECTORY`.
Then run `worldmot/scripts/analyze_vggt_point_pilot.py --predictions NEW_DIRECTORY
--output NEW_DIAGNOSTIC.json --render` offline. CUDA_VISIBLE_DEVICES was 5 for
the pilot, with both GMT and VGGT belonging to this method on that one V100.
