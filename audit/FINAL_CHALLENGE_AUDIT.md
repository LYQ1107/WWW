# FINAL CHALLENGE AUDIT

Status: **complete and stopped**. This is a diagnosis-only audit on branch `challenge-audit`; no formal Stage1/Stage2 training, fine-tuning, threshold search, GT edit, new model module, DIVOTrack run, or WILDTRACK run was started.

## 1. Baseline reproduction

The fixed released-code single-GPU reference is the completed Stage2 run, not the paper-text recipe.

| Metric | Paper | Released-code baseline | Δ (released − paper) |
|---|---:|---:|---:|
| CVMA | 75.200 | 75.630 | +0.430 |
| CVIDF1 | 81.300 | 78.740 | −2.560 |
| MOTA | 78.000 | 80.350 | +2.350 |
| HOTA | 66.200 | 66.645 | +0.445 |
| IDF1 | 82.100 | 81.201 | −0.899 |
| AssA | 69.400 | 67.706 | −1.694 |

The baseline uses source commit `dfa9ca8e0b8da5c2af89ef3d3ac4f9d991162ebb`, Stage2 `model_20000.pth` SHA256 `d1ab611670aa0284b910b25fd251f84b4cb13ba406d7ab89226adcd6ca702e17`, 44 MOT files, and test annotation SHA256 `7a1e735bf25026a56148deaeb432e0520c6ad5dad97d5362aa58739255389593`. Full provenance is in [`baseline_provenance.json`](manifests/baseline_provenance.json).

## 2. Code-level motivation

`run_global_tracker_plus` forms `asso_nonk` from the association head, builds `id_inds` as an observation-by-global-ID membership matrix, and computes `traj_score = asso_nonk @ id_inds`. Historical observations therefore contribute uniformly. With the released threshold rule, `overlap_thresh * id_inds[:, j].sum()` grows with the number of observations assigned to ID `j`; there is no score, area, blur, occlusion, view, age, or feature-consistency weight.

`memory_bank` averages the most recent `BANK_SIZE=10` ReID features with equal weight. It has no detector-confidence, box-size, blur, occlusion, view, age, or feature-consistency term. `_forward_asso` applies random ROI-center jitter during inference, so released inference is stochastic. The audit-only switch measures this effect without changing the default path. Details are in [`CODE_AUDIT.md`](CODE_AUDIT.md).

## 3. A0: inference stochasticity

The automatically selected subset is `00001garden`, `00003garden`, and `00005garden` (annotation-count 25th/50th/75th percentile nearest scenes; ties by scene name). It contains 7,130 images and 54,372 annotations. All runs use seed-specific inference, `TEST_LEN=40`, bank on, and bank size 10.

| Condition | HOTA mean | AssA mean | IDF1 mean | MOTA mean | Runtime |
|---|---:|---:|---:|---:|---:|
| Official jitter, seeds 20260930/20261001/20261002 | 82.924 (SD 0.208) | 81.715 (SD 0.378) | 82.229 (SD 0.401) | 79.486 (SD 0.120) | 965–1112 s |
| Fixed seed 20260930, no jitter | 82.278 | 79.806 | 80.613 | 80.227 | 1823 s |

The no-jitter delta is −0.790 HOTA, −2.119 AssA, and −2.021 IDF1, with +0.605 MOTA. This is marked **RELEASE-CODE INFERENCE STOCHASTICITY ISSUE**. No-jitter is not proposed as a method.

## 4. A1: observation quality

The full 22-scene baseline observation table has 504,727 Hungarian-IoU≥0.5 diagnostic matches. Detector score shows the clearest quality relationship: dominant identity correctness is 0.8543 in Q1 versus 0.9553 in Q4 (−0.1011), and cross-view consistency is 0.8379 versus 0.9357. The scene-level score gap is negative in 15/22 scenes and its bootstrap CI excludes zero.

Area has a pooled Q1/Q4 identity gap of −0.1170, but the scene bootstrap CI includes zero and only 14 scenes have both global bins represented. The Laplacian-variance blur proxy runs in the opposite direction here: Q1 has 0.9353 identity correctness versus 0.8490 in Q4; its scene CI includes zero. These are diagnostic labels, not official HOTA/IDF1 matching.

## 5. A2: negative historical evidence

All settings are fixed seed 20260930, `TEST_LEN=40`, memory bank off, and the same six-view subset.

| Policy | HOTA | AssA | IDF1 | MOTA |
|---|---:|---:|---:|---:|
| Uniform history | 82.780 | 81.487 | 81.946 | 79.416 |
| Score top 75% | 81.827 | 79.808 | 81.130 | 79.120 |
| Score top 50% | 80.636 | 77.690 | 79.523 | 78.984 |
| Area top 75% | 81.662 | 79.361 | 80.406 | 79.361 |
| Area top 50% | 80.734 | 77.920 | 79.204 | 78.711 |

Every predeclared mask lowers the three identity-focused metrics. H2 is therefore **not supported** by this intervention. The official MATLAB diagnostics were run only for uniform and the best mask (score top 75%) on the three-scene subset: CVIDF1 81.015→80.300 and CVMA 74.891→73.507. These numbers are subset diagnostics, not full-test official results.

## 6. A3: history length and memory bank

| TEST_LEN | Bank | HOTA | AssA | IDF1 | MOTA |
|---:|---|---:|---:|---:|---:|
| 8 | on (10) | 77.487 | 71.446 | 73.805 | 79.918 |
| 16 | on (10) | 79.089 | 74.108 | 76.444 | 79.865 |
| 24 | on (10) | 80.658 | 77.068 | 79.106 | 79.824 |
| 40 | on (10) | 83.068 | 81.924 | 82.634 | 79.622 |
| 40 | off | 82.780 | 81.487 | 81.946 | 79.416 |

Longer history improves every identity metric in this fixed subset. The predeclared result is **LONG HISTORY HELPS GMT**; H3 (saturation or decline) is not supported. Plots are [`A3_history_vs_AssA.png`](figures/A3_history_vs_AssA.png) and [`A3_history_vs_IDF1.png`](figures/A3_history_vs_IDF1.png).

## 7. A4: identity-error propagation

Using the baseline observation table, there are 16,173 GT-centric wrong-identity streaks and 19,800 prediction-centric contamination events. Streak probabilities are 0.0514 for length ≥2, 0.0259 for ≥5, 0.0184 for ≥10, and 0.0130 for ≥20; the mean streak is 3.70 observations (median 1, p95 2).

The pooled cross-view conditional error rate remains much higher than the arbitrary-target baseline at every tested lag:

| Lag | P(error at t+k | error at t) | Baseline P(error at t+k) |
|---:|---:|---:|
| 1 | 0.5390 | 0.0998 |
| 2 | 0.5389 | 0.0998 |
| 5 | 0.5374 | 0.0994 |
| 10 | 0.5335 | 0.0989 |
| 20 | 0.5270 | 0.0977 |

This supports H4 as descriptive propagation evidence. The estimate uses diagnostic Hungarian matches, majority predicted-ID labels, and exact frame-plus-lag cross-view pairs; it is not a causal intervention and is not an official metric.

## 8. A5: cross-view entity retrieval

The fixed feature dump contains three float16 scene files and no image pixels. Features are aggregated by per-frame L2 normalization, mean pooling, and final L2 normalization, with a minimum of five matched observations per tracklet.

For GT-aligned tracklets, all eligible queries give appearance/fused R@1 0.8846, R@5 1.000, and mAP 0.9359. For predicted-track protocol, R@1 is 0.8889, R@5 1.000, and mAP 0.9383. Q4 gives R@1 0.9231 and mAP 0.9615 in both protocols. Q1 has zero eligible cross-view queries after the fixed positive and observation-count filter, so a low-versus-high quality retrieval effect cannot be estimated. H5 is therefore **indeterminate**, not positive evidence.

## 9. Evidence matrix

The complete matrix is [`CHALLENGE_EVIDENCE_MATRIX.md`](CHALLENGE_EVIDENCE_MATRIX.md), with machine-readable [`CHALLENGE_EVIDENCE_MATRIX.csv`](results/CHALLENGE_EVIDENCE_MATRIX.csv).

## 10. GO / NO-GO

The gate has two supported H1–H4 findings: H1 for detector-score quality and H4 for persistent cross-view error. H2 is negative, H3 is negative, and H5 is indeterminate. This is a **conditional GO to define a research challenge**, because the stated gate requires two H1–H4 findings and one of them is H4. It is **not a strong GO for the specific claim that reliability-based history filtering will improve GMT**; the fixed interventions all hurt, and long history helps.

## 11. Recommended next challenge (no method implemented)

**Reliable Persistent Identity Association under Heterogeneous Multi-View Evidence**

Core question: How can a global multi-camera tracker preserve a stable entity identity when observations across views and time have highly unequal reliability, without allowing noisy or incorrect observations to contaminate long-term identity evidence?

中文表述：当不同摄像头、不同时间提供的视觉观测质量高度不一致时，如何让全局多摄像头跟踪器长期保持稳定身份，并避免低质量或错误观测污染后续 Global Identity？

This is a challenge framing for the next explicitly approved stage; no reliability estimator, memory update, or new network was implemented here.

## 12. What this audit does not claim

- It is a released-code reproduction, not an exact paper-text 14k/Adam/batch-8 reproduction.
- The A2 MATLAB values are three-scene diagnostic subset values; the full-test baseline values remain the immutable reference.
- Hungarian matches, A1 quality labels, A4 persistence, and A5 retrieval are diagnostic analyses, not replacements for TrackEval, HOTA/IDF1, CVMA, or CVIDF1.
- H2 was not demonstrated; no claim is made that score/area filtering improves GMT.
- H5 was not demonstrated because the fixed Q1 retrieval stratum had no eligible cross-view queries.
- No test-GT threshold, keep ratio, TEST_LEN, or BANK_SIZE was selected from outcomes; all audit values were fixed before execution.

## 13. Raw result paths

- Baseline: [`baseline_provenance.json`](manifests/baseline_provenance.json), [`baseline_prediction_manifest.json`](manifests/baseline_prediction_manifest.json)
- Subset: [`sanity_subset.json`](manifests/sanity_subset.json), [`sanity_subset_test.json`](cache/sanity_subset_test.json)
- A0: [`A0_seed_variance.csv`](results/A0_seed_variance.csv), [`A0_jitter_ablation.csv`](results/A0_jitter_ablation.csv)
- A1: [`A1_quality_stratification.csv`](results/A1_quality_stratification.csv), [`A1_scene_effects.csv`](results/A1_scene_effects.csv)
- A2/A3: [`A2_history_sweep.csv`](results/A2_history_sweep.csv), [`A3_history_sweep.csv`](results/A3_history_sweep.csv)
- A4: [`A4_error_survival.csv`](results/A4_error_survival.csv), [`A4_contamination_events.csv`](results/A4_contamination_events.csv), [`A4_cross_view_cascade.csv`](results/A4_cross_view_cascade.csv)
- A5: [`A5_entity_retrieval.csv`](results/A5_entity_retrieval.csv), [`A5_entity_retrieval.json`](manifests/A5_entity_retrieval.json)

## 14. Exact commands

All inference commands were executed from `/data3/liuyeqiang/GMT_challenge_audit` with `CUDA_VISIBLE_DEVICES=0`, `GMT_AUDIT_TEST_JSON=audit/cache/sanity_subset_test.json`, and the fixed checkpoint link. Representative commands were:

```bash
./audit_tools/run_seed_sweep.py --jitter --seeds 20260930 20261001 20261002
GMT_AUDIT_DISABLE_TEST_JITTER=1 ./audit_tools/run_audit_inference.sh A0_seed_20260930_nojitter SEED 20260930 INPUT.VIDEO.TEST_LEN 40 MODEL.ASSO_HEAD.WITH_BANK True MODEL.ASSO_HEAD.BANK_SIZE 10
python3 audit_tools/run_history_sweep.py --subset --matrix all
GMT_AUDIT_DUMP_FEATURES=1 ./audit_tools/run_audit_inference.sh A5_feature_dump SEED 20260930 INPUT.VIDEO.TEST_LEN 40 MODEL.ASSO_HEAD.WITH_BANK True MODEL.ASSO_HEAD.BANK_SIZE 10
```

The official CV adapter used the installed MATLAB Engine environment:

```bash
source /data3/liuyeqiang/matlab_runtime_setup/use_matlab_r2020a.sh
python audit_tools/run_official_cv_audit.py --run audit/runs/A2_uniform_bankoff --output audit/runs/A2_uniform_bankoff/cv_official
python audit_tools/run_official_cv_audit.py --run audit/runs/A2_score_top75 --output audit/runs/A2_score_top75/cv_official
```

`python` in those two commands is the R2020a Engine Python (`py37_matlab_engine`), not the GMT conda interpreter. Assembly commands are `python audit_tools/assemble_a0.py`, `python audit_tools/assemble_a2_a3.py`, and `python audit_tools/evaluate_entity_retrieval.py ...` with the GMT environment where pandas/torch are needed.

## 15. Provenance and stop point

Audit branch: `challenge-audit`; base/source commit: `dfa9ca8e0b8da5c2af89ef3d3ac4f9d991162ebb`. Stage2 SHA256 is `d1ab611670aa0284b910b25fd251f84b4cb13ba406d7ab89226adcd6ca702e17`. The exact code/report commit is recorded after the final audit commit is created. All formal training remains paused; the worktree stops here and waits for an explicit next instruction.
