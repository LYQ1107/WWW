# Geometry recovery: recorded gate diagnostics

These are logged gate values on fixed screening cases. Repeated mask/feature/geometry variants are not independent samples. GT identity labels and downstream scores do not choose these gates. Threshold violations are not evidence of low physical camera overlap, low texture, rapid rotation or GT pose error. Missing matrix/residual-stage details remain unknown.

| Profile | Cohort | Rows | Unique scene/frames | Accepted |
|---|---|---:|---:|---:|
| neural_F_H | association_train | 60 | 15 | 0 |
| neural_F_H | strict_dev | 120 | 30 | 0 |
| plane_H | strict_dev | 60 | 30 | 0 |
| traditional_F | strict_dev | 120 | 30 | 0 |
| vggt_pose | strict_dev | 60 | 30 | 0 |

Traditional F/plane H: at least20 fit inliers, ratio≥.25, finite median≤3 original pixels, footprint≥1%. VGGT pose: at least20 finite independent background residuals, ratio≥.25 within12 original pixels, median≤12, footprint≥1%. Neural candidates additionally require20 fit inliers and independent pixel inlier support; missing support/matrix fields cannot be reconstructed as a physical cause.

Mask labels differ by profile: traditional F masks keypoints, VGGT masks the model input while its pixel reference remains background-masked, and neural matching excludes detector footprints and padding. Homography is a soft plane approximation and does not establish arbitrary 3D propagation.

The separate complete primary dense causal run records3 numerical/background gate passes, at14324 actual VGGT calls. Its full B5/B7 output attribution is independently audited; screening rejection counts must not replace full-run validity counts or GT pose accuracy.

Some early diagnostics captured dirty source trees. Matching a recorded file SHA to a later committed Git blob recovers those exact file bytes; it does not make the original whole tree clean. The earliest traditional reference only records a source commit, so that weaker provenance is explicit. JSON contains source snapshots, per-row violations and unlogged fields.
