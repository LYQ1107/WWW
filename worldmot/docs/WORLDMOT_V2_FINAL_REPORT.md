# WorldMOT v2 interim report

M0 data/source/resource audit, M1 real RGB/object-feature interfaces, M2
joint PWQT interfaces, and a finite M3 real-feature backward pilot are
implemented on the isolated branch
`research/worldmot-persistent-world-queries-20261022`.  Twenty-four focused
tests pass under the WorldMOT package.

The real pilot extracts three consecutive Scene 8 pairs with RTMDet-Ins-tiny
and Ω-512, creates multi-view assignment/visibility targets, and runs finite
cache training.  Its checkpoint and cache SHA are recorded in
`results/worldmot_v2_training_manifest.json` and remain outside GitHub.

The subsequent causal replay exposed a concrete negative result: the finite
pilot checkpoint loads, but its assignment/null logits produce no accepted
runtime associations.  This is retained as a failure case for the next
training iteration; it is not converted into a tracking score.

With explicit CUDA synchronization, the three-pair end-to-end engineering
smoke (RGB decode + RTMDet + Ω + PWQT) is 0.722 pair/s at the median and
0.270 pair/s at the inverse P95 latency.  The PWQT-only diagnostic is much
faster, so Ω and detector execution dominate; these values are not formal
long-video FPS claims.

The current evidence is an engineering milestone, not a scientific victory:
there is no full TRAIN checkpoint, no legal independent TEST score, and no
claim that PWQT currently beats the v1 Hungarian baseline.  The decoder's
query coordinates are clip-normalized; projection outputs are therefore
marked diagnostic-only.  The next gate is finite-cache training with real
detector/Ω features, followed by registered validation and official TrackEval.

