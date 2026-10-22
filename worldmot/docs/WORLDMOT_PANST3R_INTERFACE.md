# PanSt3R interface

`worldmot/models/panst3r_backbone.py` is the only model-specific adapter. It performs the official resize and `[-1,1]` normalization, preserves the original-to-model homography, and returns a `GeometryFrame` containing:

- one pointmap and confidence map per camera;
- panoptic masks and `query_id` metadata from `standard_v2`;
- shared query features;
- focal and camera-to-world matrices produced by official pointmap postprocessing;
- finite-tensor, query-shape, segment-count, and model-time diagnostics.

The adapter fails loudly for a missing checkpoint, missing CUDA, missing official dependency, or failed SigLIP vocabulary load. A valid forward with zero person segments is reported as `NO_PERSON_DETECTIONS`, which can later be compared with the independent fallback path.
