# WorldMOT v2 — Full TRAIN → Independent TEST

Status: IN_PROGRESS. This report is updated from actual artifacts; neither a configuration nor a pilot is a completion claim.

## Frozen registration (2026-10-22)

Base commit: 177adbe1391bef5af152db2eff38d1a508d98db8. Branch: research/worldmot-v2-fulltrain-test-20261022. Independent from GMT/JEV.

All 24 TRAIN scenes and 28,754 synchronized timestamps must be cached and consumed in final training. Frozen official RTMDet-Ins-tiny and Ω 512 supply object features. No GT 3D positions are model inputs. Ambiguous duplicate annotation keys are masked in identity losses; original GT is immutable.

Internal model selection: TRAIN scenes [4, 8, 12, 16, 20, 24] are held out, the other 18 are training scenes. Final fitting restarts at a fixed initialization on all 24 TRAIN scenes with seeds [20261022, 20261023, 20261024]. Clip curriculum is 8/16/32, TBPTT=8. Candidate thresholds are registered as assignment margins [0.0, 0.5, 1.0], birth probabilities [0.3, 0.5, 0.7], pair probabilities [0.5, 0.7]; selection uses TRAIN validation only. Threshold zero is not a birth setting. Positive ID delivery and continuity are audited independently from losses.

Registered methods: v1 Rule/Hungarian; v2 joint PWQT; v2 without Ω geometry/local features; v2 without appearance; v2 learned two-stage association. GMT native baseline must be run on the same independent TEST or marked incomparable with a concrete reason. Prior five-scene TRAIN scores are not TEST baselines.

TEST model and threshold selection are prohibited. TEST images must use metadata-only loading; GT is confined to posthoc official TrackEval and VisionTrack MATLAB conversion. Invalid raw-GT sequences are explicitly blocked for evaluation, never silently repaired. All TEST image sequences still receive continuous causal inference and submission files.

Cache policy: four V100 workers, complete scene timelines, immutable resumable object shards, FP16 source features, no dense pointcloud or attention caches. Large assets go under /data1/liuyeqiang/nas/A100/WorldMOT/fulltrain_20261022. The original pilot and negative results remain on their old branch.

## Actual execution

- New isolated worktree created at /data1/liuyeqiang/WWW_fulltrain from the exact specified commit.
- TRAIN source audit: 24 scenes, 28,754 synchronized timestamps, 57,508 images, 596,749 annotations.
- TEST source audit: 22 scenes, 29,019 synchronized timestamps, 58,038 images, 580,178 annotations present; raw per-camera GT files exist. The row-by-row GT version audit is complete.
- Temporal, lifecycle, feature, coordinate, and current detection-box output corrections have passed 32-frame regression and 27 unit tests. No full-training checkpoint or TEST result is claimed yet.

## Phase 1 regression (completed, not final training)

Three visual branches are retained as RGB128 / RTMDet128 / Omega256 and projected independently. Clips are causal [B,T,N,D], with one initialization and TBPTT=8. Newborn queries initialize from assigned actual object features and world observations; learned cross-view grouping enforces one observation per camera. Geometry coordinates/covariances use a persistent scene normalization. Current assigned detector boxes are evaluation outputs; pure 3D projection is a separate diagnostic.

Scene 8, consecutive frames 1-32: 206 actual observations. Before temporal birth calibration, 0 assignments / 0 births (preserved). After six regression epochs with fixed initialization and unchanged nonzero thresholds (margin .5, birth .5, pair .7), 134 assignments / 4 births, longest ID 64 observations. This checks executable persistence only and is not a full-train or independent-TEST result. Regression checkpoint SHA and curves are in results/worldmot_v2_temporal_regression32.json.

The four real cache workers process complete assigned TRAIN scene timelines on V100 GPUs 0,2,3,4. First object shards have been written and checksummed on NAS. Full training is still pending completion of all 28,754 timestamps.

## Official converter and inference entry (implemented and regression-tested)

- The exact raw-GT box multisets were compared with split annotations. TEST has 14 valid scenes, not 13: Scene 1 camera 1 has a legitimate empty final frame (GT ends at 1004, images at 1005). An empty final frame is not treated as a version mismatch. Original GT remains immutable. Eight TEST scenes contain duplicate GT identities and are blocked from official metrics, while all 22 still receive inference.
- The format-only adapter calls unchanged repository TrackEval HOTA/CLEAR/Identity and native VisionTrack MATLAB/MEX. A synthetic perfect-trajectory format regression returns 100 for expected metrics and 0 IDSW; this is explicitly not a model score. Checks are saved in results/worldmot_v2_official_converter_regression.json.
- The RGB-to-ID entry is metadata-only with an explicit --split. TEST refuses pilot/subset weights and requires a SHA-matched final TRAIN-selection lock. A complete 32-frame real RGB regression has delivered nonzero IDs and current-assigned detection boxes; pure 3D projections remain separate.
- A separate frozen native GMT comparator has passed a 32-frame TRAIN interface check. It uses the original native association method, disables CIR/GEO interventions and future-dependent short-track-count filtering, and never enters WorldMOT's model modules. Prior five-scene TRAIN guard scores will not be substituted for TEST. Native GMT's complete-video return latency and its throughput are recorded separately.
- Final epoch selection is restricted to 4 or 6 so the refit includes all 8/16/32 clip lengths; epoch 2 is retained as a diagnostic curve point. The registered TRAIN-only threshold grid remains unchanged. No independent TEST metric has been read for selection.
- A durable driver is running while caches finish. It checks all 24 scene timelines and checksums, preserves per-camera missed GT and FP supervision separately, then runs internal training, TRAIN-only selection, three full-TRAIN joint seeds and matched-budget ablations, freezes checkpoints, and runs complete RGB TEST inference plus official evaluation. This is orchestration in progress, not a completion claim.
