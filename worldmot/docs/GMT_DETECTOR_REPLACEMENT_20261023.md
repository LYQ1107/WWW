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

### Current object-cache verification Gate (2026-10-23)

The new GMT detector-feature cache has now passed the full immutable TRAIN
verification. `cache_verification/coverage.json` reports
`VERIFIED_COMPLETE_24_TRAIN_OBJECT_CACHE`: all 28,754 synchronized TRAIN
timestamps (57,508 images) were scanned, with 557,002 identity targets checked
and zero identity-label mismatches. The verifier also records finite feature,
geometry, and score arrays, `raw_GT_modified=false`, and
`GT_xyz_in_inputs=false`; the retained detector/GT failure samples are
`detection_supervision_failures.jsonl.gz` (its SHA is captured in
`results/gmt_object_cache_verification_20261023.json`).

The registered source, budget, registration, and coverage SHA256 values are
recorded in that result file. The fixed TRAIN-only queue has consequently
started the preregistered four-epoch `joint` and `no_geometry` internal fits on
the 18 TRAIN scenes, with Scenes 4/8/12/16/20/24 held out for validation.
No TEST frame or threshold was used for this transition; this Gate is cache
admission evidence, not a GMT association or tracking result.

## Full-video live/cache numeric boundary correction

The first six-scene V1 validation attempt stopped at Scene 4 timestamp 598
after 597 delivered timestamps. The strict checker found a box-width delta
of 0.0000076294 pixels: native FP32 xyxy coordinates were subtracted as Python
float64, whereas the immutable cache stores xywh as FP32. Real inference on
timestamps 597/598/599 confirms identical candidate counts, scores and every
ROI feature; FP32 boxes are also exact. This is a representation mismatch.
`live_difference_scene4_598_v1.json` retains both values and the failed
validation stream is preserved, not promoted to a completed result.

The adapter now explicitly emits FP32 xywh at its boundary, matching the
existing immutable schema. Equality remains exact, without an epsilon or
candidate substitution. Shared raw caches and training object arrays remain
unchanged. A real repeat checks the failure timestamps before fresh complete
validation. The idle fit controller is re-registered before any formal fit.
The v2 real repeat passes exact all-field equality at all three timestamps.
The complete CPU suite passes 48 tests, including the real failing coordinate.

`run_gmt_v1_queue.py --root EXPERIMENT_ROOT --gpus 0 2 6` registers the fixed
existing V1 rules with this corrected source. Three independent workers run
the six complete TRAIN validation videos into `train_v1_validation_shared_v2`.
The controller then runs official TRAIN evaluation, deeply verifies all 22
GT-free TEST candidate scenes, writes `v1_fixed_rule_test_lock_v2.json`, and
runs all 22 independent TEST videos plus TrackEval/MATLAB. Every transition
checks the source stamp. No quality threshold is selected from TEST. These
accuracy runs share GPUs with cache/training workers and are explicitly
marked `shared_gpu`; their FPS is not an exclusive speed comparison.

### Final association freeze and independent TEST admission (2026-10-23)

The registered association queue completed all four fixed-budget full-TRAIN
fits and wrote `final_test_lock.json` with four SHA-locked checkpoints:
three `joint` seeds (20261022/23/24) and one `no_geometry` seed (20261022).
All four cover 24 TRAIN scenes and 28,754 synchronized timestamps; the
selection file was frozen from TRAIN validation only (`selected_epochs=4`,
`TEST_used_for_selection=false`). The immutable freeze evidence is recorded
in `results/gmt_association_final_freeze_20261023.json`.

Independent TEST inference has now started on the complete 22-video TEST
metadata with the `gmt_pre_gta` detector in `live_verified` mode. The first
wave contains `joint_seed20261022` and `no_geometry_seed20261022`; the second
wave will run the two remaining joint seeds. Because other registered V1 and
VGGT-point jobs share the six authorized cards, the TEST route is explicitly
`shared_gpu`; its per-scene timing includes RGB decode, detector, Omega,
query/memory, post-processing and flushed ID delivery. No formal metric is
reported until all 22 scenes of a route are complete and evaluated.

### Preserved 2D-control source-guard failure (2026-10-23)

The first finite 2D-control queue reached a frozen TEST lock and produced all
22 `iou_motion` prediction manifests (29,019 frames), but stopped before the
`appearance_motion` route and before official evaluation because its registered
source SHA no longer matched the live worktree. The exact failure and hashes
are preserved in `results/gmt_2d_control_failure_20261023.json`. The 22
`iou_motion` manifests are retained as failed/partial evidence only; neither
route contributes a final metric until both controls are rerun under one
immutable current-source registration.

As a separate diagnostic, the preserved `iou_motion` predictions were later
run through the official evaluator without changing GT. The legal 14-scene
scope produced TrackEval HOTA 57.65895, AssA 53.15544, DetA 63.08694, IDF1
64.33262, MOTA 75.67827, IDSW 860, Frag 1560; MATLAB produced CVIDF1
43.96730 and CVMA 18.79811. These values are recorded in
`results/gmt_2d_iou_motion_partial_official_metrics_20261023.json` only as
partial A0 evidence: the source registration is stale and A1 is absent, so
they are not a final 2D-control or GMT comparison.

### VGGT-point TRAIN selection Gate (2026-10-23)

The finite TRAIN-only scan completed both registered partitions (324 windows
per GPU partition, six scenes, temporal and joint methods). The selector chose
the following settings without reading TEST frames:

| Method | Points/person | Minimum confidence | Minimum support |
|---|---:|---:|---:|
| temporal | 16 | 0.3 | 0.2 |
| joint | 32 | 0.2 | 0.1 |

The immutable selection and partition hashes, source SHA, and the explicit
`TEST_used_for_selection=false` flag are recorded in
`results/gmt_point_train_selection_20261023.json`. Complete six-scene TRAIN
full-video validation is now running for both settings. No TEST lock or formal
MOT metric exists at this Gate; the selection is not a final GMT comparison.

### GMT V1 official TRAIN validation Gate completed (2026-10-23)

The corrected fixed-rule V1 queue completed all six registered validation
videos (scenes 4/8/12/16/20/24; 5,860 synchronized timestamps) and ran the
official TrackEval and VisionTrack MATLAB evaluators. Only scenes 4/8/12 are
in the legal aggregate: raw GT for scenes 16, 20 and 24 contains duplicate
keys and is therefore explicitly blocked without editing GT. The complete
coverage, source stamps, evaluator hashes and metrics are recorded in
`results/gmt_v1_train_gate_complete_20261023.json`; the compact synchronized
copy is `results/gmt_v1_train_validation_metrics_v2.json`.

The valid-scope aggregate is TrackEval HOTA 48.27676, AssA 34.10812, DetA
69.06127, IDF1 49.17423, MOTA 76.17674, IDSW 665, Frag 180; the interleaved
MATLAB aggregate is CVIDF1 48.00064 and CVMA 70.27261. These are TRAIN-only
validation values, not TEST results and not a claim of improvement over GMT.
`TEST_used_for_selection=false` remains explicit.

### GMT V1 official full-TEST Gate completed (2026-10-23)

The fixed-rule GMT pre-GTA V1 route completed all 22 registered TEST videos
(29,019 synchronized timestamps) under the frozen lock. The official evaluator
used the original GT without modification, `tracks_2d_by_camera`, and no
TEST-derived threshold or checkpoint selection. Eight scenes (5, 12, 13, 14,
17, 20, 21, 22) remain explicitly blocked because their raw GT contains
duplicate keys; their inference is retained, but they do not enter the legal
aggregate. The raw evaluator output is retained at
`results/gmt_v1_test_official_raw_metrics_20261023.json` and the audited gate
summary at `results/gmt_v1_test_gate_complete_20261023.json`.

On the 14-scene valid scope, TrackEval reports HOTA **48.48305**, AssA
**41.56007**, DetA **56.87552**, IDF1 **53.07887**, MOTA **67.83044**, IDSW
**2,761**, and Frag **1,655**. The interleaved VisionTrack MATLAB evaluator
reports CVIDF1 **52.52115**, CVMA **65.28177**, IDTP 87,391, IDFP 55,608,
IDFN 102,394 and CLEAR-IDSW 7,628. These are the official GMT V1 detector
replacement numbers, not a GMT-native historical baseline and not evidence of
an improvement by the unfinished VGGT point routes.

The shared-GPU full-video timing is stored per scene in
`results/gmt_v1_test_fps_20261023.csv`: mean 0.72091, median 0.80882 and
minimum 0.26183 synchronized dual-camera timestamps/s. It is an accuracy-run
throughput record, not an exclusive latency ranking; a separate exclusive FPS
audit is still required for a fair speed claim.
