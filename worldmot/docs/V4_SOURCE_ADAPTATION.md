# V4 source adaptation and provenance

Source review is independent of measurement results. No third-party detector,
association manager, birth policy, Transformer or point tracker was imported.

| Source | Reviewed immutable revision | Actual V4 input and adaptation | License |
|---|---|---|---|
| [Stone Soup](https://github.com/dstl/Stone-Soup/tree/7ce1f1fac857969766c12f4ca1ffd594acdf5f6a) | `7ce1f1fac857969766c12f4ca1ffd594acdf5f6a` | `ConstantVelocity` / `CombinedLinearGaussianTransitionModel` in `stonesoup/models/transition/linear.py`, `KalmanPredictor` in `stonesoup/predictor/kalman.py`, `KalmanUpdater` in `stonesoup/updater/kalman.py`: six-dimensional world center/velocity, F(dt), integrated white-acceleration Q(dt), innovation covariance and Joseph update. NumPy reimplementation with `[xyz,vxyz]` order; no full framework dependency. | MIT, complete notice in `../third_party_notices/Stone-Soup-LICENSE.txt` |
| [MeMOTR](https://github.com/MCG-NJU/MeMOTR/blob/eb7a177b9cbcb89742ec69b2545ab3af2ea31a80/models/query_updater.py) | `eb7a177b9cbcb89742ec69b2545ab3af2ea31a80` | `QueryUpdater.update_tracks_embedding()`: confidence-conditioned long-memory EMA and separate recent evidence. Adapted as confidence-gated 1024-D GMT ROI long-term/per-camera short-term memory. No copied neural layers or pretrained MeMOTR weights. | MIT, complete notice in `../third_party_notices/MeMOTR-LICENSE.txt` |
| [VGGT-Ω pose encoding](https://github.com/facebookresearch/vggt-omega/blob/48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6/vggt_omega/utils/pose_enc.py) | `48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6` | `encoding_to_camera()` and `extri_intri_to_pose_encoding()`: camera-from-world extrinsics in OpenCV coordinates, intrinsics and original/model pixel transform. Reuses the existing verified Ω512 adapter, camera-forward depth and persistent-world Sim(3) contract. Local world-Z is never camera depth. | FAIR Noncommercial Research License v1 (2024-10-16); full upstream notice retained alongside source and in third-party notices |

The existing Ω512 weight SHA is
`c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`.
It is the original 512 checkpoint; no 416 reproduction is claimed. No model
weights, videos or perception caches are redistributed by this branch.

V4's CV and memory code is an adaptation of the above mechanisms, not a claim
that Kalman filtering, EMA or the upstream models are original contributions.
Stone Soup and MeMOTR's MIT status was checked from their actual LICENSE files.
Ω retains its separate research terms; it is not relabelled MIT or Apache.

CoTracker and MVTracker are not dependencies of this version. Their point
trajectories do not supply human Global IDs and no second large model was added.

## Coordinate and uncertainty limitations

VisionTrack COCO metadata has `frame_id` but no verified physical frame rate.
Default time is one dataset frame-id tick; optional verified FPS converts ticks
to seconds. Spatial scale is relative world units, never assumed meters.
Measurement covariance uses observed foreground scatter and quality, inflated
for unreliable alignment/visibility when supplied. It does not claim calibrated
frequentist coverage. The fixed first-version noise prior is registered before
outcomes; calibration claims require an additional TRAIN validation.

Two views from one Ω reconstruction share errors. C4 uses a convex mixture of
scatter plus view disagreement, performs one update per object/timestamp, and
never divides its noise by the number of cameras. Disagreement also reduces
confidence. C0 remains the frozen R2 path; all changes require explicit V4 flags.
