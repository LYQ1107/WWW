# VGGT point association

Status: IMPLEMENTING. Base: `9a5572ebb3184b58253d448648195d175983ca1c`.

Official VGGT source is already present at
`/data1/liuyeqiang/WWW_vggt_geo_runtime/sources/vggt`, clean commit
`a288dd0f14786c93483e45524328726ab7b1b4ce`. Candidate original weights are
`/data1/liuyeqiang/WWW_vggt_geo_runtime/weights/vggt_1b.pt` (5,026,874,952 bytes).
Weight hash, official provenance and strict aggregator/TrackHead loading are
component admission gates. No new weights are trained in route A.

Use official pad preprocessing at 518, preserving per-image rounded resize
scales and white padding. Query source is input zero, as required by
`BaseTrackerPredictor`'s feature sampling. Return inverse-mapped original
pixels, visibility and confidence, rejecting padding/out-of-frame values.
Run Aggregator and TrackHead only; do not compute unused geometry heads.

Temporal anchors are `[A(t-1), A(t), B(t)]` and the separately anchored
`[B(t-1), B(t), A(t)]`. Synchronous newborn association uses both `[A(t),B(t)]`
and `[B(t),A(t)]`. No timestamp later than t may be read.
Deterministic internal-box points are confidence-weighted into candidate
support. Exclusive per-camera assignment has a NULL option. Bidirectional
cross-view evidence is required to merge newborns before allocating IDs.
Historical image references and finite gap limits are recorded. Current
unmatched detections receive new IDs; unobserved predictions are not emitted.

TRAIN validation sweeps 8/16/32 points and preregistered support/rejection
settings before TEST locking. A2 is temporal only and A3 is joint temporal
and cross-view. Shared detector caches and official scoring match route B.
