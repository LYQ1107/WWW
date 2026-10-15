# Active teacher capacity before training

Observed native GMT feature width is1152; four VGGT concatenated frame/global
layer widths are2048. With128-dimensional query/decoder, a VGGT track/global QD
teacher consists of query312320 + AG278531 + QD1532165 =2123016 parameters,
about6.2% above the initial two-million target. The large fixed input dimensions
explain the difference; no backbone parameter is trained. NULL is the explicit
fixed-zero logit, allowing learned candidate scores to represent unmatched.

For an active-parameter-matched DINO control, its four1024-channel layers use
1024->228->128 projections and the same128-dimensional downstream model; the
228 intermediate width is chosen analytically to match the2048->128 parameter
budget, before any teacher results. Its key projection uses the same bottleneck.
This is an explicitly stronger nonlinear projection control, not an unused
parameter bank or duplicated feature padding. Expected total2126488 parameters,
within0.2% of VGGT. Record actual construction counts in every training manifest.
Native inputs, task/candidate sets, events, seed, optimizer, steps and ROI/crop
conditions must match. Encoder FLOPs and actual time remain different and measured.

Global and track-conditioned See-Query share an active conditioning MLP; the
global control feeds a learned global token, the proposed version feeds the
observed history. No unused MLP is counted as matched active capacity. Conditional
concat/See-Query modules are instantiated only when used. Last-layer-only has one
projection and therefore fewer parameters; it is a simple feature-depth control,
not the sole capacity attribution test. Four-layer global versus track and the
matched DINO/ordinary MLP/GMT controls carry the novelty attribution burden.

## Actual executed population and upstream budget

The parameter matching above does not imply identical effective gradient
populations. The frozen V3 task has3000 queries, but full-image visual AG/QD
trains on2201 eligible queries (2194 positive,7 genuineNULL); ROI/nativeMLP uses
2733 (2721 positive,12 genuineNULL). The267 masked queries are not negatives.
Among supervised queries,532 have no raw patch center and are excluded from
full visual gradient fitting; this is distinct from the565 zero-center queries
in the complete fit index. Both VGGT and DINO visual eligible-key order hashes
match exactly. Native/ROI are strong same-task controls, not strict identical-
gradient-population controls. The crop comparisons separately preserve the same
234 supervised queries/256 fit across every variant, with full-image fallback
draws retained. See trackvggt_fit_population_disclosure_v1.json.

Every formal no-cycle phase totals1000 updates (150 pilot plus850 resumed),
but QD additionally reuses a1000-update AG query/sampler initialization for
both VGGT and DINO. NativeMLP has no upstream AG pretraining. No equal-total-
upstream-optimization claim is made against nativeMLP, and shared pretraining
cost is not counted repeatedly as new GPU work. These differences limit
capacity-only/intrinsic-feature attribution; they do not alter completed fits
or the independently matched DINO/crop/cycle evidence.
