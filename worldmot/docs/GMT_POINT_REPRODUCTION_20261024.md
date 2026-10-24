# Frozen source and result reproduction

The tested detector/geometry runtime and point/control runtime have different
frozen source fingerprints. Both code versions are published. Run each route
from its matching source pin so the existing guards pass. This audit compared
every contributing Python source file and all eight provenance scripts by
exact Git blob SHA against the remote trees; it also checked the separate
inference/cache/worker drivers. No guard was disabled and no GPU experiment
was repeated.

| Route | Published source pin | Verified runtime SHA256 |
|---|---|---|
| GMTDet+V1 and GMTDet+V2 | `373631187ab9790132c073b3a66d0a84e860367d` on `research/worldmot-gmt-detector-20261023` | `c0ef87cc85759f654156c9ec71208bca12bc4f120cb0174264682b9015859e2e` |
| VGGT point A2/A3 and 2D A0/A1 | `12057c4e784a2e98cd38c40b420f7d2c53f91008` on `research/worldmot-gmt-point-comparison-20261023` | `0d6b8bc62da81fa4423afd808ea6fc55d05face2ab57c32e7d49eeb69256a229` |

The full source-file SHA maps, local checkout HEADs and additional driver
blob checks are in `results/gmt_point_reproduction_sources_20261024.json`.
Detector runtime checks cover 55 package Python files and eight provenance
scripts; point/control checks cover 63 package files and eight scripts.
GitHub commits were published through the Git Data API, so their commit IDs
differ from local Git HEADs even where all tested source bytes are identical.
The older local point-development worktree is not the frozen TEST source.

## CPU checks on the existing server

The clean detector worktree and current comparison runtime satisfy the
fingerprints above:

```bash
cd /data1/liuyeqiang/WWW_gmt_detector_20261023/worldmot
PYTHONPATH=.:scripts CUDA_VISIBLE_DEVICES='' /data1/liuyeqiang/envs/worldmot311/bin/python -c \
  "from experiment_provenance import source_stamp; assert source_stamp()['sha256']=='c0ef87cc85759f654156c9ec71208bca12bc4f120cb0174264682b9015859e2e'"

cd /data1/liuyeqiang/WWW_gmt_point_comparison_20261023/worldmot
PYTHONPATH=.:scripts CUDA_VISIBLE_DEVICES='' /data1/liuyeqiang/envs/worldmot311/bin/python -c \
  "from experiment_provenance import source_stamp; assert source_stamp()['sha256']=='0d6b8bc62da81fa4423afd808ea6fc55d05face2ab57c32e7d49eeb69256a229'"
```

For a fresh checkout, use the exact remote commits above in separate detached
worktrees after fetching those commits. Do not switch branches inside a live
experiment worktree. The data, environments and external frozen official
model checkouts referenced below are the existing server assets; their SHAs
are recorded in the route Gates and inference manifests.

## Existing predictions and official evaluation

All eight completed routes use the same immutable raw detection cache:
`/data1/liuyeqiang/nas/A100/WorldMOT/gmt_detector_comparison_20261023/raw_test_v1`.
The raw RGB and GT root is `/data/DATASETS/TRACKING/JDE/VisionTrack`.
Metadata and original-GT audit are under
`/data1/liuyeqiang/nas/A100/WorldMOT/fulltrain_20261022/audit/`.

Prediction roots are:

- V1: `gmt_detector_comparison_20261023/test_gmt_v1_v2`
- V2 primary: `gmt_detector_comparison_20261023/test_gmt_v2/joint_seed20261022`
- V2 no-geometry: `gmt_detector_comparison_20261023/test_gmt_v2/no_geometry_seed20261022`
- V2 already-completed additional seed: `gmt_detector_comparison_20261023/test_gmt_v2/joint_seed20261023`
- A0/A1: `final_comparison_20261023/2d_controls_v2/test/{iou_motion,appearance_motion}`
- A2/A3: `vggt_point_association_20261023/test_v1/{temporal,joint}`

Each path is relative to `/data1/liuyeqiang/nas/A100/WorldMOT/`. Individual
`scene_XX/manifest.json` files contain frames, source/checkpoint/lock SHAs and
prediction SHA; the raw files were independently rehashed for all 176 new
route/video combinations. The official evidence inventory supplies absolute
metric/manifest paths, sizes and SHAs. The full TEST attribution artifact
supplies absolute track paths, sizes and verified track SHAs.

An optional official re-evaluation of the existing primary V2 output is:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' /data1/liuyeqiang/envs/worldmot311/bin/python \
  scripts/evaluate_fullvideo.py --split test \
  --audit /data1/liuyeqiang/nas/A100/WorldMOT/fulltrain_20261022/audit/data_audit.json \
  --predictions /data1/liuyeqiang/nas/A100/WorldMOT/gmt_detector_comparison_20261023/test_gmt_v2/joint_seed20261022 \
  --box-source tracks_2d_by_camera \
  --output /data1/liuyeqiang/nas/A100/WorldMOT/final_comparison_20261023/reproduce_v2_seed22_official
```

This example is documentation; this reporting audit read the completed
official outputs instead of rerunning evaluation. Use a fresh output directory
when reproducing. Eight original-GT-invalid scenes remain inference-only;
the same 14 legal scenes enter TrackEval and the native MATLAB adapters.
`v1_observed_bbox` remains a separately labelled historical output-policy
control with its actual alternative box-source key.

## Frozen inference command templates

These templates document the original finite routes. They were not executed
in this CPU reporting Gate. The user's current policy stops new seed work.
Do not invoke the older training/test queue controllers to reproduce a single
method because those historical registrations include additional seeds.

From the detector runtime pin, V1 uses `scripts/run_fullvideo.py` with:

```text
--split test --method v1 --detector gmt_pre_gta
--metadata /data1/liuyeqiang/nas/A100/WorldMOT/fulltrain_20261022/audit/test_metadata.json
--gmt-raw-cache /data1/liuyeqiang/nas/A100/WorldMOT/gmt_detector_comparison_20261023/raw_test_v1
--gmt-detector-mode live_verified --v1-output-boxes current_detector
--lock /data1/liuyeqiang/nas/A100/WorldMOT/gmt_detector_comparison_20261023/v1_fixed_rule_test_lock_v2.json
--output NEW_EMPTY_PREDICTION_DIRECTORY --resource-mode shared_gpu
```

V2 uses the same detector/metadata/cache arguments, `--method v2`,
`--checkpoint .../gmt_detector_comparison_20261023/final/joint_seed20261022.frozen.pt`
and `--lock .../gmt_detector_comparison_20261023/final_test_lock.json`.
Use the matched no-geometry checkpoint for that separate ablation.

From the point/control runtime pin, A2/A3 use
`scripts/run_vggt_point_fullvideo.py` with:

```text
--split test --method temporal_OR_joint
--metadata /data1/liuyeqiang/nas/A100/WorldMOT/fulltrain_20261022/audit/test_metadata.json
--raw-cache /data1/liuyeqiang/nas/A100/WorldMOT/gmt_detector_comparison_20261023/raw_test_v1
--lock /data1/liuyeqiang/nas/A100/WorldMOT/vggt_point_association_20261023/point_final_TEST_lock_v1.json
--detector-mode live_verified --constant-position-cache
--output NEW_EMPTY_PREDICTION_DIRECTORY --resource-mode shared_gpu
```

A0/A1 use `scripts/run_gmt_2d_fullvideo.py` with the same split/metadata/raw
cache and live mode, `--method iou_motion` or `appearance_motion`, and lock
`.../final_comparison_20261023/2d_controls_v2/frozen_test_lock.json`.
All real measured runs record physical visible devices and precision. A new
reproduction requires normal GPU occupancy admission; do not use another
owner's process or call aggregate multi-card throughput single-video FPS.

CPU-only table, error, memory and illustration commands are documented in
`GMT_POINT_REPORTING_AUDIT_20261024.md`. The additive policy
`configs/gmt_point_effect_gate_policy_20261024.json` preserves the historical
registrations while recording the user's stop on new multi-seed experiments.
