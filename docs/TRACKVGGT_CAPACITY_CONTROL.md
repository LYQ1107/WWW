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
