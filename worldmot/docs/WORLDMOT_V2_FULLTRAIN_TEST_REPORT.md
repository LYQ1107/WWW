# WorldMOT v2 — Full TRAIN → Independent TEST

Status: IN_PROGRESS. This report is updated from actual artifacts; neither a configuration nor a pilot is a completion claim.

## Detector input correction and cache invalidation

The detector audit found that the pipeline passed RGB arrays unchanged to MMDetection's ndarray loader. Official RTMDet filename inference decodes BGR, and this checkpoint has `bgr_to_rgb=False`. The adapter now explicitly converts caller RGB to contiguous BGR. Six real images (both cameras of TRAIN Scenes 8, 9, 24) reproduce official filename inference exactly: zero box-coordinate error, zero score error, identical masks. The previous input differs on all six. Evidence is in `results/worldmot_v2_detector_input_verification.json`.

All 9,001 timestamps generated before this correction have been moved to a separate NAS negative-results directory. They are invalid for formal training; their counts are excluded from current coverage. Prior 32-frame persistence runs remain execution diagnostics produced with the old detector input, and cannot establish detector quality or a scientific tracking score. No final training or independent TEST had started before discovery. The corrected four-GPU full cache has restarted. Nothing in the raw dataset or old branch was changed.

A corrected-input Scene 10 regression uses 32 consecutive real timestamps and 64 detected observations. Cache replay and a fresh RGB run both deliver 50 assigned observations, one newborn global ID, and that ID appears at all 32 timestamps. The subset checkpoint is explicitly forbidden for TEST. Its concurrent-load RGB throughput (0.908 pairs/s) is an execution diagnostic, not a formal speed or model-quality result. The checkpoint SHA and curves are saved in `results/worldmot_v2_corrected_regression32.json`.

## Frozen registration (2026-10-22)

Base commit: 177adbe1391bef5af152db2eff38d1a508d98db8. Branch: research/worldmot-v2-fulltrain-test-20261022. Independent from GMT/JEV.

All 24 TRAIN scenes and 28,754 synchronized timestamps must be cached and consumed in final training. Frozen official RTMDet-Ins-tiny and Ω 512 supply object features. No GT 3D positions are model inputs. Ambiguous duplicate annotation keys are masked in identity losses; original GT is immutable.

Internal model selection: TRAIN scenes [4, 8, 12, 16, 20, 24] are held out, the other 18 are training scenes. Final fitting restarts at a fixed initialization on all 24 TRAIN scenes with seeds [20261022, 20261023, 20261024]. Clip curriculum is 8/16/32, TBPTT=8. Candidate thresholds are registered as assignment margins [0.0, 0.5, 1.0], birth probabilities [0.3, 0.5, 0.7], pair probabilities [0.5, 0.7]; selection uses TRAIN validation only. Threshold zero is not a birth setting. Positive ID delivery and continuity are audited independently from losses.

The epoch schedule is a fixed prefix [8,16,32,32,32,32], so a selected four-epoch final refit matches the first four epochs of the internal trial. Query miss tolerance is fixed at 64 timestamps before TRAIN selection and independent TEST. Checkpoints include optimizer state, Python/NumPy/Torch/CUDA RNG streams, source digests and cache-manifest digests; resume rejects incompatible source or cache assets.

Registered methods: v1 Rule/Hungarian; v2 joint PWQT; v2 without Ω geometry/local features; v2 without appearance; v2 learned two-stage association. GMT native baseline must be run on the same independent TEST or marked incomparable with a concrete reason. Prior five-scene TRAIN scores are not TEST baselines.

Two controls were registered before internal training. First, `no_geometry` keeps all RGB, detector and Ω local visual features while disabling geometry and reliability inputs; joint versus this control isolates geometry, whereas removing Ω local features and geometry together would confound their effects. Second, v1 additionally exports its actual Hungarian/birth IDs on current assigned detection boxes. This preserves native v1 projection output and makes its box-source advantage separately measurable. Interrupted inference files are archived, never silently overwritten.

TEST model and threshold selection are prohibited. TEST images must use metadata-only loading; GT is confined to posthoc official TrackEval and VisionTrack MATLAB conversion. Invalid raw-GT sequences are explicitly blocked for evaluation, never silently repaired. All TEST image sequences still receive continuous causal inference and submission files.

Cache policy: four V100 workers, complete scene timelines, immutable resumable object shards, FP16 source features, no dense pointcloud or attention caches. Large assets go under /data1/liuyeqiang/nas/A100/WorldMOT/fulltrain_20261022. The original pilot and negative results remain on their old branch.

## Actual execution

- New isolated worktree created at /data1/liuyeqiang/WWW_fulltrain from the exact specified commit.
- TRAIN source audit: 24 scenes, 28,754 synchronized timestamps, 57,508 images, 596,749 annotations.
- TEST source audit: 22 scenes, 29,019 synchronized timestamps, 58,038 images, 580,178 annotations present; raw per-camera GT files exist. The row-by-row GT version audit is complete.
- Temporal, lifecycle, feature, coordinate, and current detection-box output corrections have passed 32-frame regression and 36 unit tests. No full-training checkpoint or TEST result is claimed yet.

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
- Threshold selection is followed by complete causal replay of all six TRAIN-validation scene timelines. That audit cannot change the selected thresholds; an empty or nonpersistent joint model blocks TEST until training is fixed. Epoch supervision counts are sums over actual valid observations, with separate ambiguous, positive-GT and FP counts.
- The delivery path additionally exports all 44 TEST camera streams from metadata and predictions without GT, including the eight locally GT-invalid scenes. Official conversion preserves source box coordinates at round-trip precision. Per-scene TrackEval metrics use the official metric combiners, not a mean of camera scores. Final FPS uses actual scene wall time through the last delivered IDs and includes telemetry gaps between frames.
- The final collector checks all seven full-TRAIN trials, three-seed persistent ID delivery, all complete TEST image streams, common official evaluation scope and native cross-view metrics before permitting a complete artifact status. Failure matching remains a separately labelled posthoc diagnostic; official IDSW is retained unchanged.

## Full TRAIN readiness audit while extraction continues

The read-only identity-capacity audit covers all 24 scenes and 28,754 TRAIN timestamps. After applying the existing duplicate-key supervision mask, the largest registered clip needs 61 identities (Scene 24, length 32); 128 slots are sufficient for teacher-forced training without dropping identities. The largest 65-timestamp GT identity window needs 62. This GT audit does not guarantee capacity under false-positive runtime births; complete TRAIN-validation streams separately retain overflow and rejection counts.

The first four complete corrected-input scenes (2, 9, 10, 11; 7,977 timestamps) have been independently checked for full time coverage, NPZ checksums, finite inputs and presence of all three visual branches. Geometry-valid flag fractions range from 97.7% to 99.0%; these flags combine successful alignment and point-support checks and do not measure reconstruction accuracy. These are partial extraction audits, not a replacement for all-TRAIN verification or official TEST results. Evidence is saved in the query-capacity and completed-scene feature audit JSON files under `results/`.

The independent TEST image preflight decoded all 58,038 images in all 22 scenes (29,019 synchronized timestamps), checked their dimensions against metadata and retained per-image byte fingerprints. There were no missing, corrupt or mismatched-size images. This 416.3-second CPU audit used metadata only: no GT, model inference, feature cache, threshold selection or accuracy measurement. Its summary and source fingerprint are in `results/worldmot_v2_test_rgb_preflight.json`; the detailed compressed image-fingerprint records remain on NAS. The complete preflight is a readiness check and does not constitute independent TEST results.

The coordinate diagnostic covers five complete TRAIN scenes (2, 3, 9, 10, 11; 9,177 timestamps). Each retains one normalization scale throughout its timeline. There are 56 alignment failures in Scene 3, retained rather than hidden. Scene 2's accepted Sim(3) reference scale ranges from 0.146 to 1.037 and ends at 0.193, starting from 1.0. Scene 10's initial normalization scale is 0.004796 in arbitrary reconstruction units; its normalized observation radius has median 102.36 and maximum 266.86. These are numerical conditioning and possible alignment-drift concerns for the registered geometry control, not proof of metric reconstruction error. No fixed-background anchor or metric GT is present in these cached statistics, so they cannot validate static-world drift. Inputs, training choices and thresholds remain unchanged. `results/worldmot_v2_coordinate_diagnostics.json` retains the evidence, and full-cache verification will report the same diagnostics for all 24 scenes.
