# GMT pre-GTA detector replacement

Status: IMPLEMENTING. Base: `9a5572ebb3184b58253d448648195d175983ca1c`.

The frozen GMT configuration uses RGB, DLA/BiFPN and CenterNet proposals,
`ROI_HEADS.NO_BOX_HEAD=True`, and an association candidate score threshold
of 0.525. The earliest native detection set is therefore the current-image
CenterNet proposals filtered at this threshold, before track ID creation.
Extracting these candidates does not require calling the GTA transformer.

The native `_forward_asso()` also jitters ROI boxes for its appearance
descriptor. This random appearance operation must not change the detector
candidate set. The independent adapter extracts its explicitly labeled frozen
appearance feature from the unjittered current boxes for reproducible A1/V2
features. Native detection parity will be tested on real images.

Required audits: official RGB preprocessing parity, invertible input/box map,
person-only/current images, both views, single image versus prefix equality,
zero GTA calls, checkpoint/config/source SHA, original candidate counts and
ROI dimensions. Test selection is forbidden. Shared immutable caches store
scene/frame/camera/image IDs and stable detection indices without GT identity.

Milestones: B-M0 adapter/cache; B-M1 TRAIN detector comparison; B-M2 V1;
B-M3 full 24-scene object cache and association training; B-M4 frozen TEST;
B-M5 official comparison/error attribution. All results remain NOT_AVAILABLE
until backed by actual completed artifacts.

## Completed component and TRAIN validation evidence

The adapter passed real native-box/score parity and single-image/prefix
equality on both cameras of Scene 8 frames 1-4. There were zero GTA calls.
An 8-timestamp candidate cache also passed exact box/score/ROI-feature replay
from the Python 3.11 caller through the persistent native Python 3.10 worker.
Checkpoint SHA is
`cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8`.

Fixed native operating-point detection evaluation on **all 5,860 synchronized
timestamps / 11,720 images** of TRAIN validation scenes 4,8,12,16,20,24:

| Detector | Threshold | Precision | Recall | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| GMT pre-GTA | 0.525 | 96.4691% | 89.9411% | 166660 | 6100 | 18639 |
| RTMDet-Ins-tiny | 0.25 | 75.4120% | 47.8545% | 88674 | 28912 | 96625 |

Both use exactly the same original TRAIN GT, fixed IoU 0.5 and descending-IoU
exclusive matching. This is fixed-point detection precision/recall, not AP or
official TEST tracking. GMT was already trained on VisionTrack TRAIN; RTMDet
uses its official COCO weights. The detector training histories differ, so
this result cannot be attributed to new geometric association. Shared GMT
candidates in the next comparisons isolate the associator.

## Real V1 and V2 interface smoke tests

On the same first 32 TRAIN Scene 8 timestamps, GMTDet+V1 (current detection
boxes with actual V1 IDs) emits all 128 candidates, produces two global IDs,
and shares the correct identity in 64/64 GT-matched cross-view events. True
RGB-to-final-ID E2E FPS is 0.590361 on a dedicated V100. These are pilot
diagnostics, not a full-video MOT ranking. Original V1 projection boxes are
retained separately in `tracks_2d_projection_by_camera`.

GMT+Omega object features have a distinct compatible schema:
`worldmot_object_v4_gmt_rgb128_detector128_omega256_geometry18_scores2`.
The 32-timestamp cache contains 128 valid observations, all matched to legal
TRAIN identities. Only the association transformer was trained in a **subset
engineering smoke test**, four epochs, seed 20261023. It ended with total
loss 0.378611 and is marked TRAIN_SUBSET_COMPLETE / formal inference forbidden.
It is not a full-TRAIN model and cannot enter TEST. Full 24-scene object feature
extraction is now assigned to three workers; official association training
waits for complete coverage and independent supervision verification.

## Commands and artifacts

The three raw candidate export workers use physical GPUs 0/2/6, eight disjoint
scenes each, prioritizing the six validation scenes. The object-feature workers
can share those same cards during cache construction (this is not a latency
benchmark). Up to six physical cards including the original ablations and
the point route are admitted. Later speed comparison needs exclusive GPU runs.

Use the existing GMT Python for `scripts/export_gmt_raw_detections.py`, with
`PYTHONPATH` set to this worktree's `worldmot` directory. For features, use
the existing worldmot311 Python and `scripts/run_gmt_feature_queue.py
--raw-cache RAW_TRAIN --output FULL_OBJECT_CACHE --scenes ... --label gpuN`.
The feature worker loads frozen Omega once and waits for any missing immutable
raw scene; its status, PID and assigned scene list are written under its output.

For live V1: `scripts/run_fullvideo.py --split train --metadata TRAIN_METADATA
--method v1 --detector gmt_pre_gta --gmt-raw-cache RAW_TRAIN --scenes 8
--limit 32 --output NEW_DIRECTORY`. The default is real detector inference
verified against common candidates. Explicit `--gmt-detector-mode replay`
is labeled association replay and never reports system E2E FPS.

All large artifacts are under
`/data1/liuyeqiang/nas/A100/WorldMOT/gmt_detector_comparison_20261023/`:
`raw_train_v1/`, `fulltrain_objects_v1/`, `train_v1_smoke32_v1/`,
`omega_objects_smoke32_v1/`, and `association_smoke32_v1.pt`.
Small result copies in `worldmot/results/` contain the measured counts,
per-scene/camera detector diagnostics, stage timings and source/weight SHA.

## Complete raw TRAIN cache gate and registered fit queue

All 24 raw GMT TRAIN scenes are now complete: 28,754 synchronized timestamps,
57,508 images, 557,002 candidate observations. Deep verification checks every
compressed data shard SHA, coordinate/score/feature finiteness, 1,024-D native
ROI shape, stable per-camera detection indices, both image IDs per timestamp,
native score threshold, original RGB fingerprints, and absence of GT/track-ID
arrays. There are zero empty ROI descriptors. Total compressed shard size is
1,041,985,391 bytes. Each sidecar SHA is included in the retained audit.

`raw_train_integrity_v1.json` and `detector_frontend_freeze.json` preserve this
admission. The detector-only freeze SHA is
`54f9e2db2befc946e8fc165a87986148c7294a9c456a2983ef9e9625dcbb085f`.
GT-free TEST detector candidates may now be exported at the same fixed native
threshold. This does not authorize association TEST before TRAIN selection.

The finite matched association budget is committed in
`configs/gmt_detector_association_budget_20261023.json`: internal joint and
no-geometry fits, four epochs, seed 20261022; final all-24 refits, joint seeds
20261022/23/24 and no-geometry seed 20261022, four epochs. The selector accepts
the explicit methods and the single preregistered epoch budget 4 while
preserving its original defaults for the old experiment. A durable controller
`run_gmt_training_queue.py --root EXPERIMENT_ROOT --gpus 2 6` waits for all
24 complete **object-feature** scenes, independently verifies supervision,
then trains, calibrates on TRAIN only and freezes new weights. It never
starts TEST automatically. Source or budget changes cause an explicit failure;
an incomplete cache or the 32-frame smoke weights cannot enter formal fitting.

Projection metadata follow-up: current bbox-only GMT evidence is now reported
as `CURRENT_BBOX_AND_PROJECTION` / `current_target_detection_bbox`, with
`native_view_mask=False`. A real nonempty mask has its own mask flag. Current
camera support is derived from the actual accepted observation-to-ID memory
map, so a historical mask from another camera cannot manufacture current
visibility. This correction changes diagnostic visibility evidence and does
not alter the primary current detection boxes. Existing pilot/validation v1
outputs retain their original code SHA. The idle training waiter was stopped
before it had any child fits, its registration snapshot was preserved, and
the source registration is renewed before formal training proceeds.
The V1 per-frame ablation label now also records GMT_PRE_GTA for this frontend;
the original RTMDet label is preserved for original RTMDet runs. Previously
saved pilot streams retain their original labels and exact source SHA, while
their detector manifests/candidate checks already identify the actual GMT input.

The first real two-frame follow-up (`train_v1_visibility_audit2_v1`) failed
the mask assertion. Its output is retained. The root cause was in
`lift_detection`: it replaced `det.mask=None` with the synthetic rectangle
used for 3D sampling. The corrected implementation preserves the actual mask
and stores that rectangle separately as `geometry_support_mask`, retaining
the exact sampling and fusion coverage calculation. The repeated real run
(`train_v1_visibility_audit2_v2`) passes all eight current projection rows:
zero false mask claims, real current bbox evidence and the GMT ablation tag.
`gmt_bbox_visibility_evidence_audit_v2.json` records the output/source hashes.

An independent rerun of the same 32 TRAIN Scene 8 timestamps verifies every
array in both object-cache shards is bit exact: 128 532-D observations,
geometry, covariance, identity targets and projection boxes. This limited
real parity check and the exact-raster unit test support keeping the existing
full-cache workers running; they do not constitute a full-cache rerun.
`gmt_geometry_sampling_parity_v2.json` records this evidence. Previously
saved diagnostics retain their original metadata and source hashes.
The detector worktree's complete CPU test suite passes 47 tests after this fix.
