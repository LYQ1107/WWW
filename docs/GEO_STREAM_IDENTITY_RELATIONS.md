# Observable identity relations across complete ablations

Every row is independently paired to its same-protocol locked baseline over all five strict scenes. Counts classify unordered pairs of common, one-to-one IoU≥0.5 GT-matched detector outputs; pairs never cross scenes. Long tracks contribute quadratically. These are output relations, not numbers of Hungarian decisions. Added/removed and unmatched detections are separately excluded from correctness counts. GT enters only post-run analysis.

Offline GMT includes its original full-video postprocessing; causal GMT uses the separate streaming baseline. Values across those protocols are not paired comparisons. Missing complete suites remain pending. Actual pre-bank Hungarian decision recording is a separate scene1 diagnostic.

| Method | Protocol | Matched detections | Added/removed output detections | Cross-camera fixes | Cross-camera harms | CVIDF1 | IDSW |
|---|---|---:|---:|---:|---:|---:|---:|
| B0 / B0_strict_suite_v1 | offline | 59749 | 0/0 | 0 | 0 | 95.631664 | 41 |
| B1 / B1_dinov2_strict_suite_v1 | offline | 59749 | 0/0 | 0 | 0 | 95.631664 | 41 |
| B1 dense / B1_dinov2_dense_strict_suite_v1 | offline | 59748 | 4/2 | 3648 | 0 | 95.633374 | 41 |
| B2 / B2_native_absolute_strict_suite_v1 | offline | 59749 | 0/0 | 0 | 0 | 95.631664 | 41 |
| B2 dense / B2_native_absolute_dense_strict_suite_v1 | offline | 59738 | 74/16 | 3373 | 9767 | 95.591173 | 43 |
| B2 row normalization / B2_native_row_strict_suite_v1 | offline | 59749 | 0/0 | 0 | 0 | 95.631664 | 41 |
| B3 / B3_multilayer_strict_suite_v1 | offline | 59749 | 0/0 | 0 | 0 | 95.631664 | 41 |
| B4 / B4_native_scalar_w2_k10_strict_v1_suite | offline | 59745 | 1/5 | 0 | 0 | 95.628245 | 41 |
| B5 / B5_lowrank_vggt_seed14_k10_strict_v1_suite | offline | 59744 | 3/8 | 0 | 0 | 95.630666 | 41 |
| B6 / B6_vggt_camera_strict_suite_v1 | offline | 59749 | 0/0 | 0 | 0 | 95.631664 | 41 |
| B7 / B7_lowrank_vggt_camera_seed14_k10_strict_v1_suite | offline | 59744 | 3/8 | 0 | 0 | 95.630666 | 41 |
| GMT head k10 / C_lowrank_gmt_seed14_k10_strict_v1_suite | offline | 59748 | 2/3 | 0 | 0 | 95.634085 | 41 |
| DINO head k10 / C_lowrank_dinov2_seed14_k10_strict_v1_suite | offline | 59748 | 2/3 | 0 | 0 | 95.634085 | 41 |
| B0 causal / causal_B0_strict_suite_v1 | causal | 60080 | 0/0 | 0 | 0 | 95.057962 | 482 |
| B7 dense / causal_B7_fixed_k1_strict_suite_v1 | causal | 60080 | 0/0 | 73939 | 5645 | 95.147448 | 381 |
| B7 k5 / causal_B7_fixed_k5_strict_suite_v1 | causal | 60080 | 0/0 | 14717 | 7386 | 95.064470 | 474 |
| B7 k10 / causal_B7_fixed_k10_strict_suite_v1 | causal | 60080 | 0/0 | 1019 | 5229 | 95.053081 | 489 |
| B7 k20 / causal_B7_fixed_k20_strict_suite_v1 | causal | 60080 | 0/0 | 1015 | 0 | 95.059589 | 482 |
| B8 / causal_B8_fixed20_total80_strict_suite_v1 | causal | 60080 | 0/0 | 1015 | 0 | 95.059589 | 482 |
| B9 / causal_B9_joint_total80_strict_suite_v1 | causal | 60080 | 0/0 | 0 | 0 | 95.057962 | 482 |
| Random / causal_random_total80_strict_suite_v1 | causal | 60080 | 0/0 | 1651 | 1673 | 95.057962 | 482 |
| Motion / causal_motion_total80_strict_suite_v1 | causal | 60080 | 0/0 | 0 | 0 | 95.057962 | 482 |
| Identity / causal_identity_total80_strict_suite_v1 | causal | 60080 | 0/0 | 3720 | 0 | 95.064470 | 474 |
| GMT dense head / C_gmt_head_dense_causal_strict_suite_v1 | causal | 60080 | 0/0 | 69792 | 7777 | 95.127924 | 405 |
| DINO dense head / C_dino_head_dense_causal_strict_suite_v1 | causal | 60080 | 0/0 | 77130 | 1824 | 95.157210 | 375 |
| Native fixed prior / native_prior_seed20261014_strict_suite_v1 | causal | 60080 | 0/0 | 104685 | 4554 | 95.181615 | 351 |
| B5 dense head seed20261014 / causal_B5_dense_seed20261014_strict_suite_v1 | causal | 60080 | 0/0 | 73939 | 5645 | 95.147448 | 381 |
| Native fixed prior / native_prior_seed20261015_strict_suite_v1 | causal | 60080 | 0/0 | 98661 | 60949 | 95.124670 | 379 |

Pending complete suites: causal_B5_dense_seed20261015_strict_suite_v1, causal_B7_dense_seed20261015_strict_suite_v1, native_prior_seed20261016_strict_suite_v1, causal_B5_dense_seed20261016_strict_suite_v1, causal_B7_dense_seed20261016_strict_suite_v1.
