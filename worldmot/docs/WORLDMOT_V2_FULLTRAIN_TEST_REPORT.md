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
- TEST source audit: 22 scenes, 29,019 synchronized timestamps, 58,038 images, 580,178 annotations present; raw per-camera GT files exist. Legality/version audit remains in progress.
- Temporal, lifecycle, feature, coordinate, and current detection-box output corrections have passed 32-frame regression and 27 unit tests. No full-training checkpoint or TEST result is claimed yet.

## Phase 1 regression (completed, not final training)

Three visual branches are retained as RGB128 / RTMDet128 / Omega256 and projected independently. Clips are causal [B,T,N,D], with one initialization and TBPTT=8. Newborn queries initialize from assigned actual object features and world observations; learned cross-view grouping enforces one observation per camera. Geometry coordinates/covariances use a persistent scene normalization. Current assigned detector boxes are evaluation outputs; pure 3D projection is a separate diagnostic.

Scene 8, consecutive frames 1-32: 206 actual observations. Before temporal birth calibration, 0 assignments / 0 births (preserved). After six regression epochs with fixed initialization and unchanged nonzero thresholds (margin .5, birth .5, pair .7), 134 assignments / 4 births, longest ID 64 observations. This checks executable persistence only and is not a full-train or independent-TEST result. Regression checkpoint SHA and curves are in results/worldmot_v2_temporal_regression32.json.

The four real cache workers process complete assigned TRAIN scene timelines on V100 GPUs 0,2,3,4. First object shards have been written and checksummed on NAS. Full training is still pending completion of all 28,754 timestamps.
