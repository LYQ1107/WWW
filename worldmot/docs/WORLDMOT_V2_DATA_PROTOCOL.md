# Data and supervision protocol

The M0 audit uses the read-only VisionTrack TRAIN annotation at
`/data/DATASETS/TRACKING/JDE/VisionTrack/annotations/train.json` and image
root `/data/DATASETS/TRACKING/JDE/VisionTrack/train`.  The annotation SHA-256
and per-scene counts are recorded in `results/worldmot_v2_data_audit.json`.

The audit found 24 scenes, 28,754 synchronized frames, 57,508 images and
596,749 annotations.  It also found 1,893 duplicate
`(scene, instance, frame, view)` keys.  This is retained as a data-quality
finding and must be resolved or explicitly handled before formal training.

Runtime code receives only RGB, detector outputs, and Ω geometry/features.
Ground truth is permitted only in target construction and post-hoc
evaluation.  Feature caches reject `gt_xyz`, `ground_truth_xyz`, and
`oracle_xyz` fields while retaining explicit `camera_ids` and valid masks.
Scene-qualified IDs are mapped to randomized clip-local
identity classes, so a fixed global class index is never learned from the
scene number.

