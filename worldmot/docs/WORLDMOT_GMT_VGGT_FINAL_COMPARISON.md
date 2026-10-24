# GMT detector replacement and VGGT point association

Status: COMPONENTS_VALIDATED_V1_TEST_COMPLETE_REMAINING_ROUTES_RUNNING. Registration date: 2026-10-23.

The user execution specification is the attachment
`72c424c4-a10d-471c-a81d-d2acf67a046b/Pasted text.txt`. It remains the
completion contract. Negative pilots do not stop the registered comparisons.

## Immutable baseline and isolation

All three new worktrees start from verified remote commit
`9a5572ebb3184b58253d448648195d175983ca1c`, tree
`f1110304b519c2561e38d06c88434989e85c51af`.

| Route | Branch | Local worktree |
|---|---|---|
| Detector replacement | `research/worldmot-gmt-detector-20261023` | `/data1/liuyeqiang/WWW_gmt_detector_20261023` |
| Pretrained point association | `research/worldmot-vggt-point-association-20261023` | `/data1/liuyeqiang/WWW_vggt_points_20261023` |
| Final comparison | `research/worldmot-gmt-point-comparison-20261023` | `/data1/liuyeqiang/WWW_gmt_point_comparison_20261023` |

The original experiment worktrees remain on their original branches.
The original ablation/evaluation queues are preserved. No-appearance and
two-stage are complete; no-geometry continues.
The capacity TEST waiter PID 20615 was identity/cwd/owner/children checked
and suspended with SIGSTOP before it started TEST. Its checkpoints remain
preserved. Pause evidence is in
`/data1/liuyeqiang/nas/A100/WorldMOT/capacity_20261022/queue_pause_20261023.json`.

## Frozen comparison contract

TRAIN covers 24 scenes / 28,754 synchronized timestamps. Internal validation
uses scenes 4, 8, 12, 16, 20, 24; fitting uses the other 18 scenes before
refitting on all 24. TEST covers all 22 videos / 29,019 synchronized timestamps.
The metadata/data SHA must be checked against the original audit before runs.
Official scoring uses original unmodified GT on scenes
1,2,3,4,6,7,8,9,10,11,15,16,18,19. Scenes
5,12,13,14,17,20,21,22 retain their raw GT block and still require full inference.
All new primary outputs use `tracks_2d_by_camera` and current detection boxes.
The existing TrackEval/MATLAB adapters are reused unchanged.

Registered systems are GMT native, original RTMDet V1 and three-seed V2;
A0 GMTDet+IoU/motion; A1 GMTDet+appearance/motion;
A2 GMTDet+VGGT temporal points; A3 GMTDet+VGGT temporal/cross-view points;
B1 GMTDet+V1; B2 GMTDet+retrained V2 joint and no-geometry.
A4 point+Omega consistency is conditional on completing A3 and resource budget.
No new capacity architecture is part of this experiment.

GMT candidates must be captured before GTA/Track Bank/track IDs and pass
single-image versus causal-prefix equality. Shared caches contain detector
observations and features, never GT identities. A2/A3 retain every legal
current candidate as matched or newborn; point failure cannot erase a box.
VGGT's actual reference image is always first. Two anchor directions have
separate forwards and their costs are both measured.

Association settings are selected only on TRAIN. Point count candidates are
8/16/32, with 16 the preregistered default. Point confidence/visibility, minimum
support, NULL and finite history thresholds will be enumerated and locked
before TEST. The V2 initial matched budget is the prior 4 epochs, with the
existing internal validation procedure; any longer TRAIN budget must be
separately registered before new TEST inference.

## Resource and storage policy

User authorizes up to six V100 cards and parallel jobs where memory permits.
Admission checks actual occupancy and ownership. Other users' jobs are not
stopped. A shared-card run is labeled as such; comparable latency measurements
require exclusive single-card conditions. Four/six-card aggregate throughput
is not reported as video FPS. Host RAM was 66 GiB available at initial audit;
local `/data1` had 98 GiB free. Large artifacts use the verified SSHFS mount
`Valadmin@127.0.0.1:/home/A100` (3.7 TiB available), under:

* `/data1/liuyeqiang/nas/A100/WorldMOT/gmt_detector_comparison_20261023/`
* `/data1/liuyeqiang/nas/A100/WorldMOT/vggt_point_association_20261023/`
* `/data1/liuyeqiang/nas/A100/WorldMOT/final_comparison_20261023/`

Each milestone must include code/tests, configs/SHA, commands, real status,
failure evidence and small results on GitHub. Unmeasured entries are
NOT_AVAILABLE, never zero. Final reports must include per-scene/per-camera
metrics, detector-to-output recall, point evidence quality, failure attribution,
stage P50/P95, GPU memory and RGB-to-final-ID E2E FPS. Cache replay has its own
`association_replay_FPS` label. Experiments can complete with
NO_DEMONSTRATED_GAIN; scientific success requires actual measured benefit.

## Results

The GMT pre-GTA detector + fixed V1 association now has a complete official
TEST Gate. All 22 videos (29,019 synchronized timestamps) were inferred under
the frozen lock; the legal raw-GT scope is 14 scenes, with eight duplicate-key
GT scenes blocked without modifying GT. TrackEval on the valid scope is HOTA
48.48305, AssA 41.56007, DetA 56.87552, IDF1 53.07887, MOTA 67.83044, IDSW
2,761 and Frag 1,655. VisionTrack MATLAB is CVIDF1 52.52115 / CVMA 65.28177.
The complete provenance is in `results/gmt_v1_test_gate_complete_20261023.json`
and the raw official report is in
`results/gmt_v1_test_official_raw_metrics_20261023.json`.

New VGGT-point TEST scores and the V2 route scores remain `NOT_AVAILABLE`.
The VGGT temporal and joint routes have now completed their registered
six-scene TRAIN full-video validation and official TRAIN evaluation; their
independent TEST inference is running under a frozen TRAIN-only lock.
Historical GMT/V2 TEST metrics are reference evidence only and are not
hyperparameter-selection data.

The GMT pre-GTA detector + fixed V1 association has now completed its
official TRAIN validation Gate. All six registered validation videos are
covered (5,860 synchronized timestamps); the legal aggregate contains only
scenes 4/8/12 because raw GT in 16/20/24 has duplicate keys. TrackEval on the
valid scope is HOTA 48.27676, AssA 34.10812, DetA 69.06127, IDF1 49.17423,
MOTA 76.17674, IDSW 665 and Frag 180. The interleaved VisionTrack MATLAB
result is CVIDF1 48.00064 / CVMA 70.27261. This is a TRAIN-only diagnostic
Gate; it is not a TEST comparison or a claim of gain over GMT. Full provenance
is in `results/gmt_v1_train_gate_complete_20261023.json`.

## Completed first gates

The GMT raw adapter passes exact native-candidate parity, causal-prefix and
cross-environment immutable replay on real TRAIN images. The VGGT-1B weight
hash matches the official author's LFS pointer and four real TrackHead
component forwards pass. The merged two-route code is tested independently
of the still-running original experiment worktree.

On all six complete TRAIN validation scenes (5,860 synchronized timestamps),
raw GMT detection precision/recall is 96.4691%/89.9411%, versus RTMDet's
75.4120%/47.8545% at their registered native thresholds. Their detector
training histories differ (VisionTrack-trained GMT versus COCO RTMDet).
These are detection diagnostics and do not measure geometric identity gain.

On a shared 32-timestamp TRAIN Scene 8 pilot, both new routes deliver all
128/128 GT-matched detections. GMTDet+V1 shares the correct ID in 64/64
eligible cross-view events, whereas point A3 at the registered default
confidence gates shares 0/64. V1 yields two IDs, point A3 four. Measured live
pilot E2E is 0.590361 FPS for V1 and 0.196039 FPS for the corrected point
version. This small pilot is not the final complete-video comparison and
does not replace independent TEST evaluation. Point temporal support has
124/124 GT-consistent positive-support candidate pairs; these are distinct
from selected assignments and are not dense point ground truth.
The first point failure and the corrected-coordinate/lifecycle repeat are
both preserved on the point branch with exact inference source SHA.

Full TRAIN raw candidate extraction and full GMT+Omega object-feature
extraction initially ran in parallel, with disjoint per-worker scene lists and the
same immutable shared detector lock. The completed V1 TRAIN validation is
marked shared-GPU and its latency cannot be promoted to the final comparable
exclusive-GPU table. The point temporal and joint full-video TRAIN gates are
now complete; their shared-GPU timing is recorded separately from final TEST
latency.

The matched-budget V2 registration is four epochs for joint/no-geometry
internal fits, seed 20261022, using the original 18/6 TRAIN split. Thresholds
use the existing fixed-window TRAIN calibration procedure and complete
validation audit. Full 24-scene refits use joint seeds 20261022/23/24 and
no-geometry seed 20261022, also four epochs. This matching budget is fixed
before new TEST; any expansion requires a separate TRAIN-only registration.
The 32-frame subset fit is an engineering check and has no TEST permission.

Published result/source branches:

* [Detector route](https://github.com/LYQ1107/WWW/tree/research/worldmot-gmt-detector-20261023)
* [Point route](https://github.com/LYQ1107/WWW/tree/research/worldmot-vggt-point-association-20261023)

The live status is generated from `/proc`-verified owned processes plus actual
scene artifacts in `results/gmt_vggt_live_status.json`. It is a timestamped
snapshot, not a substitute for checking current process handles. The original
official baseline results, including source metric SHA and GT blocks, are in
`results/gmt_vggt_original_official_baselines.json`. New final TEST rows remain
NOT_AVAILABLE until the respective complete inference and official scores exist.

The complete raw GMT TRAIN gate has now passed deep verification of every
data shard and all 24 timelines: 557,002 observations / 57,508 RGB images,
1,041,985,391 compressed bytes, zero empty ROI descriptors. The independently
frozen detector now exports GT-free TEST candidates concurrently; new TEST
association has not started. The registered V2 training controller is live
and waits for complete 24-scene object-feature coverage and independent GT
supervision verification before fitting. Its actual stage and process handle
are included in the generated live snapshot. No formal new association
checkpoint or final metric is claimed while that gate is pending.
# Six-card execution and additional real evidence

The current allocation stays within physical V100 cards **0/2/3/4/6/7**.
Cards 0/2/6 run disjoint GMT+Omega object-cache partitions and fixed-rule V1
validation concurrently. Card 4 continues the original no-geometry TEST.
Cards 3/7 run disjoint TRAIN point-calibration scenes. The 2D controls first
calibrate on CPU/cache replay, then may share 0/2/6 for real detector TEST.
Full point TEST can use these same six cards after TRAIN selection, complete
validation and freezing; each worker waits for at least 16 GB free GPU memory
and 16 GB available host RAM. No other user's process is stopped. Concurrent
accuracy runs are labeled `shared_gpu`; combined card throughput is never
reported as one video's FPS.

Two genuine engineering failures are retained. The first V1 full validation
stopped at TRAIN Scene 4 timestamp 598, because Python float64 xyxy subtraction
and the immutable FP32 xywh cache differed by 0.0000076294 pixels. Candidates,
scores and ROI features were otherwise exact. The corrected adapter emits the
declared FP32 schema and real timestamps 597/598/599 pass strict equality.
The first visibility follow-up also failed: synthetic bbox sampling support
had been assigned to `det.mask`. That rectangle is now separate from a real
mask; all eight current projection rows in a real two-frame repeat pass.
Every array of a 32-timestamp, 128-observation object-cache rerun is bit exact.
Original failed streams, numeric evidence and source hashes remain available.

Both default point methods now cover all 523 TRAIN Scene 8 timestamps:

| Method | Shared candidates / outputs | Selected temporal identity consistency | Correct shared cross-view IDs | E2E FPS, unoptimized |
|---|---:|---:|---:|---:|
| A2 temporal | 1999 / 1999 | 1949 / 1961 | 0 / 936 (disabled by design) | 0.217704 |
| A3 joint | 1999 / 1999 | 1949 / 1961 | 0 / 936 | 0.174433 |

These fixed-IoU post-hoc TRAIN diagnostics are not official TEST scores or
dense point ground truth. Positive-support candidate edges are now reported
separately from selected ID transfers. The default A3 gate has demonstrated
no cross-view identity benefit on this complete scene.

A model-only audit found official TrackHead recomputes the same image-independent
sinusoidal grid in every refinement iteration. Process-local caching of this
104,109,712-byte constant preserves genuine pretrained xy/visibility/confidence
outputs exactly. The paired three-image, 32-query-anchor benchmark has a 5.7387x
warm model-only speedup. A separate controlled 32-timestamp real MOT run on
exclusive GPU 3 includes RGB decode, live GMT detection, both anchored forwards,
memory and flushed final IDs: **0.191291 -> 0.594831 FPS (3.10956x)**. Every
delivered ID, point-evidence value and association decision is bit exact.
Median paired VGGT model time is 4.50912 -> 0.970596 s; the difference from the
component benchmark reflects its different query count/input setting. This is
a speed result on a short TRAIN prefix, not full-video quality or TEST FPS.

The finite point budget is registered in
`configs/gmt_point_train_budget_20261023.json`: 18 settings per A2/A3, six
validation scenes, 32-frame start/middle/end windows (648 independent windows;
20,736 processed timestamps across settings). Exact-request replay keys contain
original RGB bytes, image order, exact FP32 queries, weights/source/precision;
there are no GT or identity labels. This replay accelerates TRAIN calibration
and never supplies system E2E FPS. The selection objective is explicitly named
`TRAIN_window_identity_consistency_F1`, with formula and tie breaks in code.
It is not an official MOT metric. Selected settings must next complete all six
validation videos and official TRAIN evaluation before a TEST lock is written.
Final point TEST uses live GMT and real forwards, retaining only the proven
image-independent constant-grid cache.

A0 uses IoU plus constant box-velocity prediction; A1 uses the independent
frozen GMT ROI descriptor plus motion and current cross-view appearance.
Neither calls GTA, VGGT, GT calibration or 3D geometry. Their preregistered
budgets are three A0 IoU settings and nine A1 IoU/cosine settings. They preserve
all current candidates with NULL/newborn decisions and run the same complete
TRAIN-validation/freeze/full-22-TEST gate. Code hashes for their independent
control module and runners are included in the lock. Already separate IDs are
not retrospectively merged in this simple 2D control.

The original two-stage ablation has also completed 22/22 TEST videos and the
official legal-GT scope: HOTA 35.19849, AssA 47.59979, IDF1 41.57418,
MOTA 31.89188, IDSW 373. Its 65,580 true positives and 124,205 misses explain
why fewer ID switches are not evidence of superiority to GMT. VGGT-point and
V2 official TEST metrics remain pending; no threshold is chosen from these
historical TEST references.
The merged CPU suite passes 79 tests, including NULL assignment, mask
semantics, box storage precision, exact model-request replay integrity and
causal 2D lifecycle tests.

The first-frame failure is rendered directly from RGB and actual forward
outputs, without GT input:

![Actual A3 IDs and A-to-B predictions](../results/figures/vggt_joint_scene8_cross_view_failure.jpg)

Dots are actual Camera A query points; crosses are actual predictions in B,
colored by the source predicted identity. Their median cross-view confidence
is 0.235 despite median visibility 0.974. The registered 0.5 gate rejects
this evidence. The full finite scan measures whether relaxing confidence also
admits wrong-person matches. To get an earlier diagnostic without adding a
new setting, the already registered K=16/confidence=0.2/support=0.1 A3 first-32
window is executed separately on shared GPU 3. Its result is not a freeze or
a substitute for all-window selection and full-video validation.

The early registered K16/c0.2/s0.1 run is complete: all 128 detections are
delivered with two IDs; 64/64 shared-GT cross-view events have the correct
shared ID, with zero observed different-person merges in this 32-frame
window. All 124 selected temporal transfers are GT consistent. This is a
short two-person TRAIN diagnostic. Its measured 0.573900 FPS is on a shared
GPU and is excluded from the main full-TEST comparison. The all-window scan
and selected complete-video validation remain required; no threshold has
been selected from TEST or promoted based on this prefix alone.

The registered all-window TRAIN scan has now completed. Its selected A2/A3
settings are temporal `points_per_person=16, minimum_confidence=0.3,
minimum_support=0.2` and joint `points_per_person=32,
minimum_confidence=0.2, minimum_support=0.1`. The complete six-scene
full-video TRAIN gate is recorded in
`results/vggt_point_train_gate_complete_20261024.json` and the compact
metrics table is `results/vggt_point_train_metrics_20261024.csv`. On the
legal raw-GT scope (scenes 4/8/12), temporal obtains HOTA 65.81177 / IDF1
69.21970 and joint obtains HOTA 66.14742 / IDF1 69.99049. Joint is slightly
better on this TRAIN diagnostic (HOTA +0.33565, IDF1 +0.77079), but it is
also slower (0.44038 versus 0.65468 harmonic six-scene FPS). This is a
TRAIN-only gate; no official TEST metric or final route ranking is implied.

The frozen point TEST lock is now valid (`TEST_used_for_selection=false`),
and the live point TEST stage starts with joint. The first-wave joint worker
needed an auditable recovery for missing scene manifests (scenes 4, 6, 10,
12, 16, 20, 21); all recovered outputs are full-video and are listed in the
JSON gate artifact. The eight raw-GT-blocked scenes remain blocked without
modifying GT.

`build_gmt_point_comparison.py` creates the required partial official main
JSON/CSV, error-attribution JSON and FPS CSV. Point and V2 TEST scores remain
`NOT_AVAILABLE` until their own complete official gates. Short TRAIN speed,
model-only throughput and historical complete TEST throughput have distinct
scope labels. Raw per-frame timing aggregation for legacy P50/P95 remains
explicitly pending.

### V1 official TEST Gate (2026-10-23)

The fixed V1 detector-replacement route completed all 22 TEST videos. Its
shared-GPU per-scene E2E timing is recorded in
`results/gmt_v1_test_fps_20261023.csv` (mean 0.72091, median 0.80882,
minimum 0.26183 synchronized dual-camera timestamps/s). Because these are
concurrent accuracy runs, they are not an exclusive single-video latency
ranking. The eight raw-GT-blocked scenes are retained as inference coverage
but excluded from the official aggregate; the new point and V2 routes cannot
be compared scientifically until their own complete official gates exist.
