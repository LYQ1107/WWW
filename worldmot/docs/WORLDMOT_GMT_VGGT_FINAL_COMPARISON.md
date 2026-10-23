# GMT detector replacement and VGGT point association

Status: COMPONENTS_VALIDATED_FULL_COMPARISON_RUNNING. Registration date: 2026-10-23.

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
Three active original ablations and their official evaluation continue.
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

New methods: NOT_AVAILABLE (implementation and component preflight underway).
Historical GMT/V2 TEST metrics are reference evidence only and are not
hyperparameter-selection data.

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
124/124 GT-consistent accepted pairs; this is not dense point ground truth.
The first point failure and the corrected-coordinate/lifecycle repeat are
both preserved on the point branch with exact inference source SHA.

Full TRAIN raw candidate extraction and full GMT+Omega object-feature
extraction run in parallel, with disjoint per-worker scene lists and the
same immutable shared detector lock. Full V1 TRAIN validation is marked
shared-GPU and its latency cannot be promoted to the final comparable
exclusive-GPU table. Point default temporal full Scene 8 runs next to the
other jobs on a separate card, followed by default joint full Scene 8.

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
