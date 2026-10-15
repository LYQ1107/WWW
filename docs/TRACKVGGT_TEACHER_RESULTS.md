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
