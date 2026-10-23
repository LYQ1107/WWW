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

## Corrected coordinate/lifecycle pilot v2

The next revision uses exact PIL pixel-center resize coordinates, preserves
separate rounded x/y scales, and permits late identity aliasing only with
bidirectional point support. It rejects merging IDs that ever coexisted in
the same camera, enforces current per-camera exclusivity, and changes only
future deliveries. Twenty-two focused tests pass, including a late merge
without changing the already delivered prefix and historical-conflict vetoes.

A second real live-detector 32-timestamp run is preserved separately at
`train_joint_smoke32_v2/scene_08/`, with inference source SHA
`7749579913d89792b869a253bd17f7e5e7faf158b34dba27dba8655a9f564a13`.
The code snapshot was independently checked against this stored SHA.
It again outputs 128/128 GT-matched boxes, keeps four IDs, and shares the
correct cross-view identity in 0/64 eligible events at the same fixed default
confidence/support thresholds. All 124 accepted temporal support pairs are
GT-identity consistent. E2E is 0.196039 synchronized timestamps/s on a V100.
The earlier negative result remains retained; the engineering fixes are not
reported as an identity improvement. Full TRAIN calibration and complete
video comparisons are still required.

The finite TRAIN-only calibration search is registered before any point TEST:
point count in {8,16,32}; minimum confidence in {0.2,0.3,0.5}; support threshold
in {0.1,0.2}; visibility fixed at 0.5; minimum valid/inside points fixed at 3;
reference gap fixed at 3; lost retention fixed at 30. Evaluate fixed 32-frame
start/middle/end windows in each of the six internal validation scenes, then
run complete validation videos with the selected temporal/joint settings.
Tie-breaking prefers higher identity consistency, then fewer wrong merges,
then higher rejection thresholds, then fewer query points. Selection must
include raw detection-to-output coverage, correct and incorrect cross-view
merges and temporal identity continuity. No TEST result may select thresholds.

Full-video defaults and this finite calibration search are separate registered
TRAIN diagnostics; cached calibration, if implemented, must be labeled replay
and cannot produce system E2E FPS. All final TEST speed runs require real RGB,
real live detector and real anchored VGGT forwards.

## Complete temporal TRAIN Scene 8 diagnostic

A2 at the registered defaults finished all 523 synchronized timestamps with
1,999 shared detections and 1,999 delivered boxes. Post-hoc raw TRAIN GT
matches 1,971 boxes. Of 1,961 **selected temporal assignments** with known
source/target identities, 1,949 are consistent (99.3881%) and 12 are wrong;
there are 15 diagnostic per-view ID changes. These are fixed-IoU TRAIN
diagnostics, not official TEST metrics. The measured exclusive V100 E2E
throughput is 0.217704 dual-camera timestamps/s.

The diagnostic script now separates all positive-support candidate edges
from selected Hungarian decisions. Earlier v1 JSON labels called all positive
edges `accepted_point_pairs`; those edges were not all selected. The preserved
v1 report has 2,552 such edges, 1,960 consistent (76.8025%). The corrected v2
report names this quantity explicitly and adds actual selected assignments.
No inference output changed. Cross-view same-ID is 0/936 for this A2 control,
which intentionally does not associate across cameras; A3 runs separately.

Evidence: `results/vggt_point_temporal_fullscene8_train_diagnostic_v2.json` and
NAS `vggt_point_association_20261023/train_temporal_default_fullscene8_v1/`.

A3 with the same defaults also finishes all 523 timestamps and 1,999 outputs.
Correct cross-view shared ID remains **0/936**; 1,949/1,961 selected temporal
assignments are GT consistent. It produces 21 IDs, as does A2, with 11 versus
15 diagnostic per-view ID changes. These counts are not official IDSW and
do not prove a cross-view improvement. Its unoptimized exclusive V100 E2E is
0.174433 timestamps/s, versus A2's 0.217704. Both use the same source hash
`7749579913d89792b869a253bd17f7e5e7faf158b34dba27dba8655a9f564a13`.
The default confidence gate still rejects useful cross-view evidence on this
complete video. The preregistered TRAIN window scan evaluates confidence
0.2/0.3/0.5 and support 0.1/0.2 at 8/16/32 points; TEST remains unopened for
selection. Full A3 evidence is in
`results/vggt_point_joint_fullscene8_train_diagnostic_v2.json`.
