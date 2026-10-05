STATUS: BASELINE_ANOMALY_UNRESOLVED

# GMT/VisionTrack baseline incident — 2026-10-05

这是一份 baseline root-cause audit 的主记录。当前 canonical Stage2 OFF 的
TrackEval 数值明显低于预期数量级；在 raw detection、映射、源码语义、配置
和训练 exposure 审计完成前，不允许把任何 JEV 相对提升解释为论文有效性。

## Immutable inputs

| Item | Value |
| --- | --- |
| canonical checkpoint | `/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth` |
| checkpoint SHA256 | `cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8` |
| inference config | `/data1/liuyeqiang/WWW_jev_v2/configs/VISION_test.yaml` |
| config SHA256 | `bdaeca71d875e8824c3eaf825967a7ba032514297642a6aabe7ad10a740be94a` |
| TEST annotation | `/data/DATASETS/TRACKING/JDE/VisionTrack/annotations/test.json` |
| TEST annotation SHA256 | `7a1e735bf25026a56148deaeb432e0520c6ad5dad97d5362aa58739255389593` |
| canonical OFF prediction | `/data1/liuyeqiang/WWW/outputs/research_final_v2/off/inference_test/inference_VISION_test/coco_instances_results.json` |
| prediction SHA256 | `a8d2f5c4cc758a887276e59671831215d36a8949d07486a34d78076aaf3e7b3d` |
| formal worktree | `/data1/liuyeqiang/WWW_jev_v2` |
| current audit worktree HEAD | `9c0546d` plus the pending diagnostic tools/evidence recorded below |
| strict traced run source commit | `f3187b8` (later commits only changed documentation/configuration of Git publication) |
| prediction-converter last source change | `8d7078c3e29bb333388cd2df78e97b95c6792374` |
| exact historical evaluation commit in manifest | `NOT_RECORDED; must not be guessed` |

The Stage2 OFF baseline artifact is a diagnostic/reporting result, not a policy
selection input. The downloaded GT is retained without deduplication or label
changes; the existing permissive TrackEval audit mode is recorded separately.

## Observed anomaly

| Metric | Current artifact |
| --- | ---: |
| HOTA | 8.3140 |
| DetA | 6.4506 |
| AssA | 11.9220 |
| IDF1 | 8.5297 |
| MOTA | -70.2413 |
| IDSW | 3,247 |
| Frag | 60,103 |
| predicted detections | 527,474 |
| GT detections | 580,178 |
| CVIDF1 | 8.1854 |

These values do not identify the cause. The possible causes remain evaluator or
conversion mapping, bbox scale, frame/view mapping, category policy, score or
short-track filtering, JEV-disabled baseline regression, inference-config
mismatch, training-budget mismatch, model failure, or multiple causes.

## Preliminary TRAIN mini-subset evidence

The fixed two-scene/all-view TRAIN subset (`00002garden`, `00004garden`, 5,636
images) was evaluated directly from raw COCO predictions and annotations. The
unmodified raw `image_id` alignment is very poor, but a deterministic remap
derived only from the source order contracts changes the result dramatically:

| Direct diagnostic | Before remap | After deterministic order remap |
| --- | ---: | ---: |
| mean max IoU | 0.0945 | 0.8243 |
| median max IoU | 0.0000 | 0.8746 |
| Recall@IoU 0.5 | 5.39% | 95.57% |
| Precision@IoU 0.5 | 5.55% | 98.39% |
| TP / FP / FN @IoU 0.5 | 991 / 16,856 / 17,402 | 17,560 / 287 / 833 |

The remap is not a tuned threshold or a model change. `GMTDatasetMapper` feeds
view-block order; `GTRRCNN.sliding_inference_GMT` returns frame-major order;
`MOTEvaluator.process` currently zips the original inputs with those reordered
outputs. On this subset, 57,460 of 57,508 image labels change under the
deterministic mapping. Evidence file:

`/data1/liuyeqiang/WWW/outputs/research_final_v2/baseline_debug/output_order_mini.json`

SHA256: `9e007a05191fa3839ceb33aecbda3eb912e98339e7fb82a402fed80dfdc8d0e3`.

This makes `EVAL_MAPPING_BUG` the leading root-cause candidate, but it is not
yet the final diagnosis. Full TRAIN evidence, COCO/MOT round-trip, corrected
OFF evaluation, and pre-JEV/source regression must still pass before the
baseline is declared recovered.

## Full TRAIN order audit

The same source-derived mapping was run on the complete canonical TRAIN
prediction file. The result is a deterministic, data-independent mapping
effect, not a mini-subset artifact:

| Direct diagnostic | Before remap | After deterministic order remap |
| --- | ---: | ---: |
| mean max IoU | 0.1258 | 0.7888 |
| median max IoU | 0.0000 | 0.8415 |
| Recall@IoU 0.5 | 10.15% | 89.91% |
| Precision@IoU 0.5 | 11.11% | 98.46% |
| F1@IoU 0.5 | 10.61% | 93.99% |
| TP / FP / FN @IoU 0.5 | 60,553 / 484,351 / 536,196 | 536,530 / 8,374 / 60,219 |

All 24 audited two-view videos report `FRAME_MAJOR_OUTPUT_VS_VIEW_BLOCK_INPUT`.
The full evidence is retained at
`/data1/liuyeqiang/WWW/outputs/research_final_v2/baseline_debug/output_order_train.json`
(SHA256:
`07ce57732fe43a52d13fd5c2e9bc235c0e260a5b9074415fdbb979baafdc9816`).

## Corrected TEST diagnostic copy

The canonical TEST prediction was copied through the same deterministic
source-order remap. This is a diagnostic copy; the canonical OFF prediction
and its original report were not overwritten. The corrected copy changed
527,026 prediction rows and 57,994 image labels. Its SHA256 is
`263f6aaedbac159e6c3c4a2703dd31b0ca0d24bfe7f351bd0807872c5af57903` and its
manifest is
`/data1/liuyeqiang/WWW/outputs/research_final_v2/baseline_debug/canonical_test_predictions_order_fixed.json.manifest.json`
with SHA256
`5fa0cd9c227bf9b0bdc16382de1125545f8a12231f345fbaffe6ff3e8fffb8f0`.

Using the corrected copy in the existing permissive diagnostic TrackEval
conversion gives:

| Metric | Corrected diagnostic value |
| --- | ---: |
| HOTA | 67.442 |
| DetA | 66.278 |
| AssA | 68.992 |
| IDF1 | 82.239 |
| MOTA | 80.942 |
| IDSW | 3,092 |
| Frag | 8,004 |

The metrics file is
`/data1/liuyeqiang/WWW/outputs/research_final_v2/baseline_debug/corrected_test_eval/evaluation/metrics.json`
(SHA256:
`dfb5e83f96f3e70fc412938ab7db694555c39efdf5d6536da52a890882e11041`).
The corresponding diagnostic cross-view values are CVIDF1 `79.0248`,
sequential CVMA `80.9276`, and interleaved CVMA `74.2925`; the cross-view file
is
`/data1/liuyeqiang/WWW/outputs/research_final_v2/baseline_debug/corrected_test_eval/crossview/metrics.json`
(SHA256:
`ef9c3ce4cdbe992fd012aa11b2be0b34e660c751f3845d2d2230de6812f789e5`).
These numbers support the mapping diagnosis, but are not yet official final
paper numbers because source regression and the formal evaluator fix remain
outstanding.

## COCO/MOT round-trip audit

The corrected prediction was checked against the prepared MOT rows for all
527,474 prediction rows and a seeded 1,000-row sample. The audit passed with
zero unknown image IDs, zero missing sequences, zero mismatches, maximum bbox
absolute error `5.0e-7`, and maximum score absolute error `5.0e-7`.

Evidence:
`/data1/liuyeqiang/WWW/outputs/research_final_v2/baseline_debug/coco_mot_roundtrip.json`
(SHA256:
`0412381b261a07d7bfd69cf0e30abcbef8e440d55afc4518b2f0eb3736937bae`).
This rules out the COCO-to-MOT numeric conversion as the primary explanation
for the original anomaly; it does not by itself validate the final evaluator
patch.

## Mandatory audit order

1. Run `reproduction_tools/diagnose_raw_detection_alignment.py` on TRAIN, with
   raw COCO boxes and annotations only. Report direct IoU, coordinate bounds,
   1560-coordinate diagnostic scaling, frame offsets, scene/view/video splits,
   category counts, and score thresholds.
2. Freeze a two-scene/all-view TRAIN mini subset in
   `manifests/baseline_debug_subset.json` and use it for all cheap diagnostics.
3. Audit COCO→MOT→COCO round-trip and image/frame/view mapping; do not silently
   alter official results.
4. Compare pre-JEV GMT source, current JEV-disabled semantics, and current
   `MODEL.JEV.MODE=off` on the same TRAIN subset/checkpoint/GPU when the live
   strict GPU0 run is no longer competing for that device.
5. Audit released/current inference configs and Stage1/Stage2 effective sample
   exposure, LR scaling, and checkpoint-load warnings.
6. Write `docs/GMT_BASELINE_ROOT_CAUSE_REPORT.md` and
   `outputs/research_final_v2/baseline_debug/DIAGNOSIS.json` only after the
   evidence chain is complete.

## Formal JEV hold

The live strict traced OFF process must not be stopped. Same-GPU OFF, 4A trace
contract, and 4B simulator checks may complete as evidence, but final policy
selection, `FINAL_SELECTION_LOCK`, official TEST, and any official JEV claim
are held until this baseline incident reaches a supported diagnosis.

No new TEST hyperparameter tuning is allowed during this audit. Old-server
checkpoints are out of scope; only `model_20000.pth`, current-run checkpoints,
the current Stage1 checkpoint, repository/config history, and current
predictions/traces/caches may be used.
