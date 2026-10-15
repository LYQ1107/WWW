# TrackVGGT evidence and remaining requirements

Status: INTERIM_REVIEWER_AUDIT. DINO temporal experiments and fresh
system measurements are still executing or waiting. This document does not
declare the full research goal complete, a successful teacher, or paper readiness.

The separate TRAIN12 engineering release is complete. Its read-only inputs are
not added to the frozen teacher fit/validation cohorts. No CV-PIL/JEV decision,
weight or algorithm enters the TrackVGGT models.

| Question | Actual evidence | Interpretation and remaining limit |
|---|---|---|
| Was the upstream mechanism inspected? | SOURCE_AUDIT plus actual official/Det backbone and attention comparisons | AG/QD attribution belongs to VGGT-Det. Identity conditioning is in added modules; the frozen VGGT backbone does not learn tracking. |
| Are histories native and causal? | Selected V3 captures, frozen observation keys and original GMT state | Histories are actual GMT tracks, including impure/unknown histories. GT labels certify supervision separately and never synthesize ideal input tracks. |
| Were full TRAIN errors audited? | Twelve complete causal scene runs,169468 primary queries,1553 certified wrong/missed | Error counts describe GMT; they are not VGGT corrections. No repeated24-video census or completed11-scene feature collection. |
| Is the held-out cohort informative? | Original300 queries,218 GT-known queries,100 pure-positive ranking cases | Native Top1 has18 certified ranking errors, but actual original primary solver has0 certified wrong/missed queries in these300. Ranking improvements cannot prove actual Hungarian/Bank repairs. |
| Were train data expanded legally? | V2 1000 fit -> V3 3000 fit across11 scenes; same original300 validation | Old1000 inputs and validation controls remain identical. Many targets/frames/persons are correlated; query count is not independent-person count. |
| Does identity AG beat ROI? | V3 AG-C R1=.54 versus matched ROI=.66 and native=.66 | Gate fails at primary seed. V2 AG-A/B/C/EMA and their fit curves remain archived; V2/V3 training populations are not interchangeable capacity controls. |
| Is AG learning active? | Finite real gradients, soft-training/hard-inference schedule, fitted endpoint diagnostics | A functioning gradient or fit R1 is not held-out identity utility. Retrospective150/1000 endpoint ranks do not reconstruct missing online Recall curves. |
| Does track-QD beat global-QD? | VGGT .25 versus.24; matched QD-capacity nativeMLP=.67 | One extra positive rank at a single seed does not establish useful conditioning or beat native. DINO track=.31 with2 known harms versus VGGT track=.25 with6 harms; full paired overlap is audited. |
| Is query conditioning responsive? | Actual per-candidate track-layer weight L1 change=.511537671 under fixed history permutation; exact permutation alignment | Aggregate means/std are invariant under permutation of the same queries. Whole-history shuffle also changes AG/decoder; it is not an isolated See-Query intervention. Track R1 improves to.39 after shuffle, failing the useful-identity test. |
| Does magnification help? | Matched224 full/local/combined:VGGT .32/.44/.41,DINO .37/.51/.46 | Both improve locally; DINO is higher in each condition. This is resolution evidence, without unique VGGT identity or3D-camera evidence. No result-driven336 escalation. |
| Are crops matched? | Same256 fit/234 supervised queries, same original300 validation, identical terminal NumPy/CPU/CUDA RNG | Full zero-center samples stay as native fallback/zero learned gradient. Direct per-step crop draw keys were not logged; code/order/seed and endpoint RNG are the available schedule proof. |
| Is the cycle causal and real? | A(t-1) -> already-processed opposite B(t) -> A(t); real pair features, absorbing NULL, GT-visible loss mask | Locked128 chains,63 visible. B<=4 and A-current is one selected detection+NULL, a bounded native subset rather than a complete scene transport matrix. No future frame or GT query construction. |
| Does VGGT cycle training help? | Same initial source/AdamW/RNG,200 extra updates,lambda0/.1 R1=.22/.23; actual minibatch/cycle hashes and terminal RNG exact | No certified wrong-ranking repairs between controls. One ambiguous->correct case was already native-correct. Training cycle loss decreases without established novel identity repair. DINO temporal control remains pending. |
| Are training/inference seeds distinct? | Actual teacher seed20261015 versus native inference seed20261014 | Pinned cycle producer generic provenance mistakenly reports14; actual settings/source/RNG prove15. Raw artifacts are retained unchanged; audit/CSV flag the correction. This is not a second trained seed. |
| Were capacity controls real? | VGGT QD2123016,DINO2126488,nativeQDMLP2132697 active parameters | DINO uses an active nonlinear projection bottleneck, not unused padding parameters. Last-layer mode is smaller; it alone cannot support capacity attribution. Frozen encoder cost is separately different. |
| Is distillation justified? | No teacher has passed the independent-relation gate in completed evidence | IRGD/KD/three-seed students are unexecuted and conditional. Do not distill poor relations merely to fill a table. Gate disposition stays provisional until remaining independent experiments finish. |
| Is full tracking verified? | Six inherited native/Prior strict5 references reverified by prediction/config/checkpoint/rawGT hashes | These are original inference-RNG baselines, not new trained teachers/students. No new learned-method final-ID metrics exist. Fresh native system replays have their own mandatory parity gates. |
| Is speed measured honestly? | Separate frozen2/4/10-stream baseline plan, real prefix gate and full replay queued | No fresh timing exists yet. Complete process wall includes imports/setup/export; synchronized profiling, local files and unequal-length independent scenes limit interpretation. No cached-head FPS or Internet latency substitution. |

The original GT files remain unchanged. Strict DEV5 omits scenes with repeated
raw GT identities rather than repairing labels. Original GMT pretraining included
development videos, and historical TEST exposure is disclosed in the inherited
scope reports; strict DEV5 is not an unseen-scene or untouched blind test.
No new TEST execution or tuning is authorized by these experiments.

The uncertainty is substantial even after all primary runs: validation uses one
held-out association scene, unknown/ambiguous cases cannot become certified
wrong links, and the primary runs use one trained seed. Poor performance at this
locked seed supports a bounded negative experiment report, not a universal claim
that VGGT lacks identity information. Any positive candidate would need matched
controls, replication and actual causal full-output evidence before deployment
or a scientific-success claim.

Results are traceable through teacher_ablation.csv and source-bound raw manifests.
The crop, QD and paired-cycle audit JSON files preserve exact known fixes, harms,
unknown transitions and checkpoint hashes. The system snapshot keeps pending
cases explicit and separates inherited full MOT references from fresh timing.
Large tokens/weights/matrices remain on the server; Git contains compact evidence.

For the remaining novelty questions: the current teachers consume frozen visual
tokens and attention; these experiments do not isolate a positive3D geometric
mechanism. Crop magnification is not a new physical camera. No result establishes
that particular layers represent viewpoint, occlusion or identity semantics.
There is no evidence to replace the stronger lightweight Native Prior with the
current teacher. Student superiority over ordinary GT-supervised MLP is untested
because the relation gate has not passed. Multiple fit scenes do not substitute
for multiple held-out evaluation scenes. Fresh local cross-camera clip retrieval
can provide an executable multi-source systems evaluation, but native-only
retrieval cannot establish TrackVGGT novelty or WWW paper readiness.
