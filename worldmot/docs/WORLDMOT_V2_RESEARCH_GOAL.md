# WorldMOT v2 research status

WorldMOT v2 studies a world-centric, persistent-query tracker for
uncalibrated synchronized multi-camera RGB.  The proposed state is a set of
world queries `Q(t)`, rather than a list of pairwise-fused detections.  Every
current camera detection remains an independent observation token and the
same query may consume at most one observation per camera.

This branch is independent of GMT.  It is based on the pinned v1 reference
commit `fbd3ac4897116bf4c5bb6c92193bc96e7691ae2a` and is intended to be
merged only after full TRAIN/validation and legal official evaluation have
been run.  Existing v1 result files are retained unchanged.

The current implementation covers the M0 audit, M1 frozen object-feature
interfaces, and an M2 CPU-smoke PWQT.  The decoder currently emits
clip-normalized query coordinates.  Until a trained calibration/projection
head is available, projections are diagnostic and are explicitly ineligible
for MOT scoring.  No random checkpoint is a formal result.

