# TrackVGGT-MOT architecture and contracts

Frozen GMT detections/association remain unchanged. Real causal history -> TrackQueryEncoder(mean/EMA, latest four observations) -> box-constrained AG-A/B/C -> independently projected VGGT layers4/11/17/23 -> QD-0 last / QD-1 mean / QD-2 concatenation control / QD-3 global See-Query / QD-4 track-conditioned See-Query -> two128-dimensional decoder layers -> candidate identity logits including unmatched. CVTIC uses only visible legal A(t-1)-B(t)-A(t) chains. IRGD is conditional on valid teacher evidence; final student accepts GMT-only signals, zero VGGT calls, bounded zero-initialized residual before unchanged GTA/Hungarian/memory.

VGGT itself stays identity-unconditioned and frozen. AG/QD are credited to VGGT-Det; proposed differences are the history identity conditioning, temporal relation supervision and relation distillation. No efficacy or geometric semantics are assumed. A current detection query never receives its GT ID. Multi-proposal tracks must be aggregated without duplicate positive/length weighting. Physical camera IDs use equality and explicit maps, not ordinal arithmetic. Empty/invalid evidence must stay neutral.

Phase0 implements the minimal query/sampler/frozen attention/See-Query/QD interfaces and tests their shapes/gradients. Actual tracker event snapshots, temporal cycle and runtime adapter are subsequent gated steps; they are not complete because these classes exist.

## Subsequent executed scope

The Phase0 interface/architecture paragraphs above describe prerequisites at
that time. Actual causal native data, all56 registered AG/QD/crop/temporal
fits and independent paired audits have since completed. The screening gate
is NO_UNIQUE_VGGT_TEACHER_EVIDENCE; IRGD/conditional trained students and
learned full-video deployment are unexecuted. Fresh native system work and
final audit are complete as separate baseline/negative evidence; see the current
teacher/system/final reports.
Implementation contracts are not a scientific-success or learned deployment claim.
