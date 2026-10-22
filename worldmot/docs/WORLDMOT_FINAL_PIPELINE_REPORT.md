# WorldMOT v0 final pipeline report

## Scope and provenance

WorldMOT is an independent RGB-only research pipeline. It does not import GMT, `gtr`, Detectron2, GMT detector/ReID/memory state, or precomputed GMT predictions. GMT integration was deliberately **not attempted**. The runtime reads two synchronized VisionTrack images at the current timestamp only; annotations are opened only by the post-hoc evaluator.

The geometry backend is the official PanSt3R v2 checkout at commit `0253909812ce570d35f2325a20a591e4785f7d7d`, with official MUSt3R at `efc5b08fbbaba814706880932ba2497472096925`. The checkpoint is `/data1/liuyeqiang/PanSt3R_official_20261021/panst3r_v2_512_5ds.pth`, MD5 `6a7a47a2ea635de05af45ded8dcabd57`. Each forward uses exactly two keyframes, `use_retrieval=False`, the `person` vocabulary, official `standard_v2` panoptic postprocessing, and official pointmap/camera postprocessing. The independent person fallback is disabled/dependency-blocked and contributed no predictions.

## Evidence for the executable loop

The real smoke test used VisionTrack Scene 8, frame 1. It produced two finite pointmaps of `288x512x3`, two confidence maps of `288x512`, `200x768` shared query features, 12 panoptic segments, and four constructed 3-D proposals. Three proposals were visible in both cameras. A reviewable point-cloud rendering from those actual pointmaps is saved as `results/worldmot_smoke_v2/pointcloud_3d.png`; the per-query rendering `pointcloud_instances_by_query.png` and per-camera overlays `view*_instances_by_query.png` show which 3-D support cluster belongs to each shared query. The companion RGB and native panoptic-mask previews are in the same directory. The official PanSt3R model timing in that smoke record was about 3.034 s; the full cold-call wall time was 28.274 s because model and vocabulary initialization are included. The machine-readable smoke record is `results/worldmot_smoke_v2.json`.

Every continuous row stores geometry status, static-background correspondence diagnostics, Sim(3) scale, persistent 3-D states, independent 3-D-to-2-D projections, and per-stage timing. Scene 8 and Scene 9 each ran 100 consecutive synchronized groups without resetting identity memory, for 200 groups total. All 200 geometry rows and all 200 alignment updates succeeded. Scene 24 was additionally run for 20 consecutive crowded-scene groups as a failure diagnostic.

The world alignment is a causal ORB/static-point correspondence followed by deterministic RANSAC Umeyama Sim(3). A failed fit leaves the last valid world frame as the reference, predicts existing tracks, and does not insert unaligned observations. A PanSt3R query index is scoped to one joint forward and is never used as a persistent ID. Projection output keeps `native_view_mask`, `projected_from_world`, and `projection_refined_by_target_view` separate; all reported runtime projections are pure world-support-point projections.

## Measured results

The 100-group engineering runs were measured on one V100 (`cuda:0`). Scene 8 took 131.747 s (`1.3175 s` mean, `1.2848 s` P50, `1.3737 s` P95), or **0.759 synchronized two-camera groups/s**. Scene 9 took 122.910 s (`1.2291 s` mean, `1.2101 s` P50, `1.3171 s` P95), or **0.814 groups/s**. Host maximum RSS was 6.08–6.14 GB. The largest observed CUDA values were about 3.95 GB allocated peak and 4.48 GB reserved peak. These are single-device latency measurements; they are not four-GPU aggregate throughput.

An additional 10-group Scene 8 run records the exact stage split after timing instrumentation: preprocessing `62.4 ms` mean, PanSt3R forward `746.7 ms` mean, panoptic postprocessing `342.2 ms` mean, pointmap/camera postprocessing `43.0 ms` mean, world alignment `196.6 ms`, instance construction `12.1 ms`, identity memory `0.62 ms`, reprojection `6.05 ms`, and full pipeline `1.4288 s` mean (`0.700 groups/s`). The v0 100-group summaries have a legacy `model_only` field from before this split; the exact forward-only number is the stage-audit run above.

Post-hoc 2-D diagnostics use IoU 0.3 and are not 3-D ground truth. Scene 8 reports box recall `0.920`, precision `0.345`, 10 ID switches, and cross-view same-global-ID accuracy `0.959` over 169 paired observations. Scene 9 reports recall `0.803`, precision `0.294`, 24 ID switches, and cross-view accuracy `0.784` over 134 pairs. The crowded Scene 24 diagnostic reports recall `0.255`, precision `0.300`, 27 ID switches, and cross-view accuracy `0.600` over 80 pairs. The evaluator retains conflict samples rather than repairing them with GT.

The low precision and the Scene 24 conflicts show that the official panoptic query output is executable but not yet a reliable person detector for this outdoor/drone data. The native route includes false positive query regions. No fallback detector was substituted and no GT mask or ID was used to improve these numbers.

## Gate status

| Gate | Status | Evidence |
|---|---|---|
| `PIPELINE_EXECUTABLE` | `PASS` | Real PanSt3R RGB smoke and continuous runs |
| `REAL_PANOPTIC_INSTANCE_STATUS` | `PASS_WITH_FALSE_POSITIVES` | Real shared queries and 3-D instances; quality remains limited |
| `WORLD_ALIGNMENT_STATUS` | `PASS` engineering gate | 200/200 long-run alignment updates succeeded; residual/scale are logged |
| `TEMPORAL_ID_STATUS` | `PARTIAL_FAILURES_RECORDED` | Persistent IDs and lifecycle work; ID switches and cross-view conflicts remain |
| `REPROJECTION_STATUS` | `PASS_WITH_DETECTION_ERRORS` | Independent world-support projection is emitted for both cameras |
| `FPS_MEASURED` | `PASS` | Per-stage and end-to-end timings, RSS, and CUDA memory recorded |
| `SCIENTIFIC_SUCCESS_NOT_YET_CLAIMED` | `true` | Detection/identity quality is not sufficient for a scientific MOT claim |

This means the engineering loop is complete enough for controlled follow-up experiments, but it is not evidence that WorldMOT has solved multi-camera MOT and it is not evidence of a GMT gain. The next research bottleneck is person-query quality and identity disambiguation, especially in crowded Scene 24; GMT should remain untouched until a fair ablation demonstrates a reliable net identity benefit.

## Reproducible artifacts

Machine-readable indexes are in `results/worldmot_final_status.json`, `worldmot_geometry_audit.json`, `worldmot_instance_audit.json`, `worldmot_projection_audit.json`, `worldmot_tracking_audit.json`, and `worldmot_fps.json`. Each 100-group scene has `tracks.jsonl`, `tracks_2d.json`, `tracks_3d.json`, and a prediction-only overlay video under `results/continuous_v0_100/scene_08/artifacts/` and `scene_09/artifacts/`. Failure samples are preserved in the three post-hoc tracking audit JSON files. Checkpoints, original images, dense pointmaps, and credential/cache files are excluded from Git.
