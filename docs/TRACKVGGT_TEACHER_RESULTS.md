# TrackVGGT teacher results

Primary seed20261015; unchanged original300 validation requests,100 with a certified pure-positive candidate. Native and NativePrior Recall@1=.66. Unknown/impure references are not certified wrong. Pair ranking is an intermediate diagnostic; none of these numbers is a full Hungarian/Bank/final-tracking correction.

|Cohort|Backbone / head|Summary|Recall@1|Shuffled|Known ranking fixes / harms|
|---|---|---|---:|---:|---:|
|V2 1000|vggt/C|mean|0.07|0.34|0 / 56|
|V2 1000|vggt/C|ema|0.14|0.46|0 / 39|
|V2 1000|vggt/A|mean|0.07|0.33|0 / 55|
|V2 1000|vggt/B|mean|0.08|0.34|0 / 55|
|V2 1000|dinov2/C|mean|0.24|0.44|0 / 20|
|V2 1000|dinov2/roi|mean|0.18|0.45|0 / 38|
|V2 1000|vggt/roi|mean|0.07|0.39|0 / 56|
|V2 1000|vggt/native MLP|mean|0.63|0.65|0 / 2|
|V3 3000|vggt/C|mean|0.54|0.47|1 / 3|
|V3 3000|vggt/roi|mean|0.66|0.56|2 / 1|
|V3 3000|dinov2/C|mean|0.46|0.48|1 / 4|
|V3 3000|vggt/native MLP|mean|0.66|0.66|1 / 0|

All settings above complete1000 total optimizer updates, including the original150-step pilots with exact optimizer/RNG continuation. V3 improves AG-C from.07 to.54 but remains below native/.66 and pooledROI/.66. DINO AG-C V3=.46; this VGGT-vs-DINO difference alone does not establish unique benefit because stronger native/ROI controls remain superior. V3 nativeMLP=.66, with one certified ranking fix and zero certified harms; uncertified transitions are separately reported in the machine audit.

The K16/24/32 ablation uses the same fitted V3 AG-C K24 checkpoint and fixed300 inputs: all three give Recall@1=.54. This is an inference-budget ablation, not independently retrained K models. CPU K24 reproduces all GPU discrete query ranks/states, with floating-point logit differences disclosed in its receipt.

The full per-size-bin and paired-key audit is results/trackvggt_ag_v2_v3_primary_results_v1.json. Native correct-to-ambiguous/uncertified transitions are separate from known harms. All primary GT-unknown requests remain in the denominator-preserving reports.

## Prespecified finite repairs and independent hypotheses

One bounded native-box224 crop comparison is fixed at256 fit queries by scene/hash round robin and the original300 validation. No new DEV12 release input/error is used. Full/local/combined variants preserve exactly the native vectors, candidate histories, scores, labels and supervised query population. Full-image zero-center draws use their genuine native fallback and exactly zero learned gradient; these draws are explicitly counted, rather than silently removed from the full branch. Both backbones have556 completed crop targets; all shared NPZ bytes1591956497 <2147483648. Actual crop extraction/probes cost.139091908GPU-wall-hours. No dense map, raw image copy or invented camera is stored. Crop fit results remain separate from full3000-fit results.

All five ordinary/track QD variants and matched global/track DINO controls are registered in manifests/trackvggt_qd_fit_v1.json. The native MLP at QD capacity2132697 parameters is separately registered. VGGT global/track QD have2123016 and matching DINO2126488 trainable parameters; lighter last-layer controls disclose their smaller capacity. Current globalQD pilot produces one constant across-query layer-weight distribution, as expected. Full fits and dynamic-query effects are not inferred from this pilot.

Actual cycle relation materialization and fitted-AG gradient checks pass on the locked128-query V3 scope:63 visible chains, B<=4, actual previous/current source keys and earlier-native score evidence. P_AB/P_BA/P_AA include explicit NULL and separate legal/visible GT masks. A_current is one actual selected detection plus NULL, a disclosed bounded subset rather than a full-scene matrix. Formal temporal benefit remains unmeasured until identical-source no-cycle trackQD1000 checkpoints receive matched200-update lambda0/.1 continuations. Both controls execute the same cycle forwards to preserve dropout consumption and record actual native minibatch/cycle-schedule hashes. No lambda.2 escalation is registered.

Completed report snapshots are generated from actual manifests into results/teacher_ablation.csv;150/1000 entries are different budgets, never interchangeable or best-iteration selection. Live QD/crop/control/cycle queue states are recorded with actual process identities in results/trackvggt_teacher_execution_snapshot_v1.json.

Status: primary AG has not demonstrated independent VGGT teacher benefit. Crop/QD/temporal hypotheses are still executing or waiting for their named dependencies. IRGD, student success, scientific success and paper readiness remain unproven.

## Completed fixed224 repair

All six same256-fit/300-validation comparisons now finish1000 updates:

|Backbone|Full|Local224|Full+local|
|---|---:|---:|---:|
|VGGT|0.32|0.44|0.41|
|Matched DINO|0.37|0.51|0.46|

These use234 common candidate-backed, GT-eligible training queries; all256 remain
in fit evaluation. The terminal NumPy draw state and CPU/CUDA Torch RNG states
match exactly across all three variants within each backbone. Full-image native
fallbacks contribute350 neutral draws during the850-step continuation; local and
combined have valid native-box tokens. Raw checkpoints are unchanged. The actual
query-order/eligibility code and terminal RNG agreement verify the matched draw
schedule; individual batch keys were not separately logged in these crop fits.
The completed audit includes all per-query known fixes/harms, ambiguous changes,
original size bins and matched DINO overlap. All300 requests are retained; all
Recall@1 values above have100 certified pure-positive ranking queries as their
denominator. All remain below native/.66; magnification helps both backbones and
does not establish unique VGGT identity information.

![Actual grid support](../results/plots/trackvggt_crop_support_v1.png)

This coordinate-only example is the first original validation key with an observed
zero-center full-image ROI. It copies no image and reads no GT. The local crop has
40 valid native-box centers,28 retained positions; the original full pool is empty.
The detector's native1152 feature remains available and is used for the original
neutral fallback; the interpolated ROIAlign control also remains defined. A denser
sampling grid is not evidence of correct identity association.

## Completed VGGT QD primary fits and temporal pair

The five no-cycle VGGT QD variants finish1000 updates: last=.25, fixed mean=.24,
concat=.23, global See-Query=.24, track-conditioned See-Query=.25. Active capacity
is1284996/2072580/2138244/2123016/2123016 respectively. The matched native-only
QD-capacity MLP has2132697 parameters and Recall@1=.67. Matched DINO global/track
and DINO cycle controls remain pending at this snapshot; they are not inferred.

Track-conditioned QD's candidate-aligned layer-weight change under history
permutation has mean L1=.511537671; global See-Query is exactly zero. The shuffled
weights match the corresponding actually permuted query weights. Means/stds over
that same candidate-query multiset are permutation invariant, so their aggregate
agreement is not evidence of an inactive conditioning module. The whole query is
permuted for AG/decoder/See-Query, while image and native pair evidence remain fixed;
this does not isolate the See-Query performance effect from the other query paths.
The track model's Recall@1 improves from.25 to.39 when history is shuffled, which
contradicts useful identity conditioning in this tested setting.

From that fixed trackQD1000 checkpoint, lambda0 and.1 each perform200 additional
updates with identical source weights/optimizer/RNG, native batch draw hashes,
cycle-key schedules and final NumPy/Torch/dropout RNG states. Initial association
and cycle losses match exactly.128 fixed cycles yield106 visible draws across200
updates. The final Recall@1 is.22/.23. The sole extra certified success at
[12,30269,2] replaces an ambiguous reference from the no-cycle continuation; native
GMT already ranks that query correctly. There are zero certified wrong-link fixes
and zero certified known harms between these two continuations. Mean observed
cycle loss falls.03238→.03082, but this is not independent identity/solver benefit.
No lambda.2 expansion is justified by this weak, unreplicated ambiguity transition;
there is no IRGD/student admission.

A reporting defect is retained and corrected explicitly: pinned0eb9af7f's cycle
fitter called generic provenance() with default native inference seed20261014,
while executed seed_all(settings.training_seed) and restored checkpoint/RNG use
training seed20261015. The raw checkpoint settings and actual source/RNG evidence
resolve15; the native inference seed remains14. The raw files/logs/checkpoint SHAs
are untouched. The paired audit and teacher CSV retain the original generic seed,
flag the correction and report the actual training seed separately. Pending DINO
continuations stay on the same pinned source and will receive the same explicit
seed-resolution audit. Future main-branch fitter metadata uses the training seed.

Saved150/1000 endpoints have also been evaluated retrospectively on at most256
scene/hash-selected original fit queries. This is not a reconstructed online
training Recall curve and does not select a best iteration. Examples: V2 AG-C
fit R1=.8257 vs retained validation.07; V3 AG-C fit=.7253 vs.54; local-crop VGGT
fit=.8541 vs.44. Original native fit values on those respective subsets are
.7844/.6652/.6652. Fit populations differ between protocols, and low loss or good
fit ranking does not establish held-out identity utility. The endpoint snapshot
marks uncompleted cases pending and preserves their exact checkpoint hashes.

The next completed DINO global-QD control gives R1=.19 (query shuffle=.42),
versus VGGT global=.24. Both share the same one known native-ranking fix
[12,30266,7], with12/6 known harms respectively. This is a single seed and
poor absolute ranking performance, not a validated VGGT benefit. The QD audit
now contains6 completed cases and1 pending DINO track-QD case; the matched
DINO temporal control is still pending. No missing result is imputed.
