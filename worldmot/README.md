# WorldMOT

WorldMOT is an independent research pipeline for synchronous multi-camera tracking in a persistent scene coordinate frame. The first geometry backend is the official PanSt3R implementation. The project does not import GMT, `gtr`, GMT detectors, GMT ReID modules, or GMT track state.

The runtime contract is:

```text
two RGB images at timestamp t
  -> one joint PanSt3R reconstruction and panoptic query set
  -> one 3-D instance per shared query
  -> causal world-frame alignment
  -> persistent WorldTrack IDs
  -> independent projection to each camera
```

The official PanSt3R and MUSt3R source trees are kept outside this repository and are selected with `--panst3r-root` and `--must3r-root`. Checkpoints and original VisionTrack images are never committed.

## Status

The deterministic geometry, world alignment, identity memory, projection, dataset adapter, and causal runtime are implemented and unit tested. The official v2 checkpoint and SigLIP vocabulary have been exercised on real VisionTrack RGB pairs. Two 100-group continuous runs (Scenes 8 and 9) and a 20-group crowded Scene 24 diagnostic are preserved under `results/`. The engineering loop is executable, but native person-query precision and persistent identity accuracy are not yet sufficient for a scientific MOT success claim. Results are written with explicit status fields; a missing person query or failed geometry is not converted into a synthetic detection.

## Quick start

```bash
cd /data1/liuyeqiang/WorldMOT
PYTHONPATH=. python scripts/audit_environment.py --output results/environment_audit.json
PYTHONPATH=. python scripts/smoke_panst3r.py \
  --panst3r-root /data1/liuyeqiang/PanSt3R_official_20261021 \
  --must3r-root /data1/liuyeqiang/MUSt3R_official_20261021 \
  --weights /data1/liuyeqiang/PanSt3R_official_20261021/panst3r_v2_512_5ds.pth \
  --scene 8 --frame 1 --device cuda:0

PYTHONPATH=. python scripts/run_pipeline.py \
  --panst3r-root /data1/liuyeqiang/PanSt3R_official_20261021 \
  --must3r-root /data1/liuyeqiang/MUSt3R_official_20261021 \
  --weights /data1/liuyeqiang/PanSt3R_official_20261021/panst3r_v2_512_5ds.pth \
  --scene 8 --limit 100 --output-dir results/continuous

PYTHONPATH=. python scripts/evaluate_pipeline.py \
  --jsonl results/continuous/scene_08/tracks.jsonl \
  --output results/worldmot_tracking_audit_scene08.json

PYTHONPATH=. python scripts/build_final_status.py --project .
```

The smoke command reports `DEPENDENCY_BLOCKED`, `WEIGHT_UNAVAILABLE`, `NO_PERSON_DETECTIONS`, or a verified tensor/geometry output instead of hiding a failure.

The final gate and all result indexes are in [`results/worldmot_final_status.json`](results/worldmot_final_status.json); the detailed interpretation is in [`docs/WORLDMOT_FINAL_PIPELINE_REPORT.md`](docs/WORLDMOT_FINAL_PIPELINE_REPORT.md). Checkpoint files, original images, dense pointmaps, and model caches remain outside Git.

The real smoke pair also has a reviewable pointmap rendering at [`results/worldmot_smoke_v2/pointcloud_3d.png`](results/worldmot_smoke_v2/pointcloud_3d.png). The per-query colored 3-D supports and camera overlays are [`pointcloud_instances_by_query.png`](results/worldmot_smoke_v2/pointcloud_instances_by_query.png) and [`view*_instances_by_query.png`](results/worldmot_smoke_v2/), alongside the two RGB and native panoptic-mask previews.
