# WorldMOT v2: larger persistent identity model

Status: REGISTERED_AND_IMPLEMENTED. Formal larger-model training has not started. Existing four-epoch weights and their independent TEST comparison remain a separate experiment.

The current evidence does not establish convergence or a capacity bottleneck. The original 953,365-parameter association model has sparse TRAIN validation checkpoints and substantial pair-loss/gradient variation. We implement a larger model and a matched training-budget control to test the capacity hypothesis. RTMDet and the verified original VGGT-Omega 512 weight variant stay frozen; both profiles read the same complete, verified object cache from all 24 TRAIN scenes. There is no GMT model dependency.

## Research and design

[MOTIP, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.html) learns identity association from object features and historical identity context. This supports keeping the visual backends frozen while increasing the learned association capacity. Its [official tutorial](https://github.com/MCG-NJU/MOTIP/blob/main/docs/TUTORIAL.md) also connects training sequence length with inference memory tolerance; model width alone does not address that temporal gap.

[MeMOTR, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Gao_MeMOTR_Long-Term_Memory-Augmented_Transformer_for_Multi-Object_Tracking_ICCV_2023_paper.html) studies long-term memory for association. Its [official training configuration](https://raw.githubusercontent.com/MCG-NJU/MeMOTR/main/configs/train_dancetrack.yaml) uses width 256, eight attention heads, six decoder layers and a learning-rate reduction. Those are useful engineering references, not evidence that these settings are optimal for VisionTrack.

The implemented PWQT-L expands WorldMOT's existing joint multiview attention and recurrent update. It is not a MOTIP/MeMOTR reproduction and uses none of their checkpoints. The first comparison changes width, depth and head count while retaining the original observation channels, geometry inputs, loss and persistent-query runtime. Geometry conditioning and alternative long-term memory remain separate hypotheses rather than unlabelled additions to the capacity comparison.

| Profile | Trainable parameters | Hidden width | Attention layers | Heads | FFN width | World queries |
|---|---:|---:|---:|---:|---:|---:|
| PWQT-S control | 953,365 | 128 | 2 | 4 | 512 | 128 |
| PWQT-L | 7,675,353 | 256 | 6 | 8 | 1,024 | 128 |

PWQT-L has 8.05 times the registered parameter count. Both consume the same 532-channel observation representation: independently projected RGB, detector ROI and Omega features, geometry and reliability. Query slot count is deliberately fixed; slot sufficiency and representational capacity are distinct questions.

## Registered finite experiment

The executable registration is `configs/worldmot_v2_capacity_study.json`. Internal fitting uses the original 18 TRAIN scenes, with scenes 4/8/12/16/20/24 held out. Both profiles use seed 20261022, AdamW, batch size four, TBPTT eight, clip lengths 8 then 16 then 32, and 24 complete epochs. Learning rate starts at 3e-4 and decreases by a factor of ten after epochs 12 and 20. Gradient clipping remains 1.0. Every two epochs is checkpointed with optimizer, RNG, source and exact scene coverage.

At registered epochs 4/8/12/16/20/24, both models receive identical TRAIN-only validation and threshold-search opportunities. The selection metric is the existing fixed start/middle/end-window cached identity-F1 surrogate; missed raw detections are outside its denominator, and it is not official IDF1 or HOTA. The earliest large-model candidate within 0.002 absolute F1 of its maximum selects a common final epoch budget for both profiles. Each profile uses its best registered thresholds at that shared epoch. Full six-scene continuous replay must then produce nonzero persistent IDs without further threshold adjustment. All candidate curves and failure records remain available.

Fresh final fits use all 24 TRAIN scenes and all 28,754 timestamps per epoch: three large-model seeds and seed 20261022 for the small matched-budget control. Final-weight TRAIN replay is a functional delivery check, not independent accuracy. Source, thresholds and weight SHAs are frozen before independent TEST. The planned TEST follow-up uses all 22 original RGB videos and the same 14 original-GT-valid scenes and official tools as the existing comparison. Capacity differences use the matched seed; only the large extension has three final seeds.

The original four-epoch model provides a short-budget reference. Its comparison with PWQT-L changes both capacity and training budget, so it cannot by itself identify a capacity effect. The new PWQT-S control and per-epoch validation curves are required. A 24-epoch limit is a finite research budget, not a guarantee of convergence or superiority.

## Implementation checks and execution

Both profiles pass real TRAIN forward/backward checks on two complete 32-frame clips from scenes 10 and 24, including the most crowded clip in each scene. The batch contains 64 synchronized timestamps, up to 46 observations per time, and all updated parameters remain finite. The CPU probe measures about 2.79 seconds for the small batch and 11.29 seconds for the large batch. These are finite training-step diagnostics, not inference FPS or a V100 training-time estimate. The source, clip coordinates and checks are retained in `results/worldmot_v2_capacity_CPU_probe.json`.

Dedicated tests check parameter scaling, the unchanged observation/identity contract, propagation of temporal gradients into all three feature branches, and exact state-dict/config restoration. Before formal fitting, the queue also executes a synchronized V100 memory/gradient probe on the assigned exclusive card. Complete-video model and end-to-end FPS will use raw RGB inference after final freezing.

The separate worktree is `/data1/liuyeqiang/WWW_capacity` on `research/worldmot-v2-capacity-20261022`. Large artifacts go to `/data1/liuyeqiang/nas/A100/WorldMOT/capacity_20261022`. The larger study starts only after all seven original final fits finish. It occupies one exclusive V100 queue slot while the other three cards run the original v1, v2 and native GMT complete-video comparisons. No four-card throughput is reported as per-stream latency. Original checkpoints are verified unchanged when the capacity training phase returns.

Known input limitations remain visible: TRAIN detector recall is approximately 55.43% under the fixed supervision-matching diagnostic, and some world-coordinate inputs have poor numerical conditioning. A larger association module cannot recover a person with no observation and is not guaranteed to repair unreliable geometry. Neither limitation proves that capacity is sufficient or insufficient.


<!-- BEGIN ACTUAL CAPACITY TRAINING -->
## Actual training evidence

Status: IN_PROGRESS.

| Trial | Profile | Seed | Completed epochs | Timestamps / epoch | Status |
|---|---|---:|---:|---:|---|

No formal fitting checkpoint is present. Configuration, unit tests and a gradient probe do not count as completed training.

Independent TEST scores and real inference FPS are pending. TRAIN optimization losses and cached identity surrogates are not independent MOT accuracy.
