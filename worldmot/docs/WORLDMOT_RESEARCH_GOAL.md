# WorldMOT research goal

WorldMOT is a separate GMT-free research line. At each causal timestamp it receives two synchronized VisionTrack RGB images and is expected to produce one joint reconstruction, shared 3-D person proposals, persistent world-coordinate IDs, and independently reprojected camera outputs.

The acceptance sequence is deliberately staged:

1. Validate official PanSt3R tensors and person vocabulary on a real pair.
2. Build one `WorldInstance` from a shared query and both pointmaps.
3. Align each new reconstruction to the previous world frame with static-background correspondences.
4. Associate proposals with an explainable 3-D memory and manage birth, loss, and deletion causally.
5. Project stored world support points into both cameras and measure real timing.

No GMT detector, ReID, track, memory state, or GT oracle is part of this pipeline. `query_id` is a per-forward proposal key and is never treated as a long-term ID.
