# Phase VII P0.5 — Reassociation Attribution

**NO-GO for expansion of the current state-only MATCH architecture.** Attribution is complete: 69 complete-video runs on preregistered TRAIN controller-heldouts 09/10/11, 11 identical-prefix events / 187 branches, 10 common H8/H16/H32 continuations. Phase VI native mechanism results are retained, not overwritten. Full24 and official TEST remain untouched.

## A. Ordinary rules with the same native solver

Fixed threshold improves pooled HOTA by +0.4066, AssA by +0.8184, compared with B2 +0.3515/+0.7083. This equals 115.68% / 115.55% of the positive B2 gain. Fixed rules beat B2 by 0.0551/0.1102. Rules trigger only 1/0/0 reassociations, B2 5/3/3: most ordinary-rule improvement cannot be attributed to frequent global re-solving. The raw score threshold and second-pass semantics are documented; GMT's track-length-scaled default threshold differs.

## B. Same solver and budget

B2 exceeds the normalized ordinary MLP by raw +0.0582/+0.1182, and under equal 5/3/3 trigger budgets by +0.0823/+0.1596. Both fail preregistered +0.1 HOTA/+0.2 AssA margins. Under the same budget B2 minus fixed rules is -0.0291/-0.0652. Dynamic threshold and unnormalized MLP collapsed on these frozen historical labels; preserve their negative results, but do not infer that their entire model families are ineffective. The LayerNorm MLP supplement was frozen from source-scale analysis before reading heldout outcomes. Three videos and one seed do not establish statistical significance.

## C. First judgment, second validation, global assignment, interactions

B2 minus VALIDATION_ONLY is -0.0065/-0.0133: global reassociation has a slightly negative complete-video effect here. The JSON provides the full global on/off × own/legacy validator factorial, plus every controller under a common legacy validator. These are policy interventions; their future states and trigger timing mutate naturally. Current-payload paired interventions instead hold the exact prefix, RNG, score matrix and candidate order fixed, then use common live B2 future policy.

All eleven B2-triggering target rows are GT-unmatched: their original target utilities are unavailable for fitting, not positive causal identity supervision. The paired audit separately measures effects on other known observations. At H32, fixed policy contributes 3 more wrong camera-frames than B2 at this enriched set, while fixed equal-current-trigger budget contributes 0; normalized MLP contributes 48. This local selected-event protection does not reverse the complete-video fixed-rule advantage. Windows overlap and are not population rates. Actual identity error episodes use fixed prefix/birth anchors, consecutive observed camera-frames and gap/end censoring; they are not a scalar proxy renamed as seconds. All full-video error episode records are published.

The mechanism journals retain score matrices, candidate semantic order, first actions/features, banned edges, second inputs/outputs, actual committed IDs, ACCEPT rows moved, solver/validator timings, corrections/regressions and segment commits. The legacy paired state digest includes identity/history/RNG/banks/metadata and omits diagnostic counters/archival assignments; a complete-field digest is implemented and clone-tested for the bounded native dataset. Complete-video inference is genuinely mutated-state and every future proposal is rebuilt. Score/candidate *functions* and exogenous perception are frozen; later score/candidate values may change as a causal consequence of changed state.

## D. Research decision

Stop expanding current MATCH on the basis of its earlier positive result. Ordinary rules explain more than all of the B2 gain in this heldout set, and Binary JEV is also stronger than B2 pooled. Candidate conditioning is a hypothesis to investigate only after bounded native learnability and MiniSet gates. No candidate training, Unified success, commercial Jev reproduction or RLCD claim is made.

## Complete pooled results

| Method | HOTA | AssA | IDF1 | IDSW | MOTA | R rows |
|---|---:|---:|---:|---:|---:|---:|
| GMT | 84.6234 | 83.7393 | 95.6684 | 341 | 97.8316 | 0 |
| FIXED | 85.0300 | 84.5577 | 96.1095 | 25 | 98.6642 | 1 |
| DYNAMIC | 2.2149 | 0.0595 | 0.0607 | 37598 | -0.3267 | 0 |
| MLP | 17.1282 | 3.4362 | 7.9660 | 34630 | 7.4880 | 8532 |
| MLP_LN | 84.9167 | 84.3294 | 95.9985 | 208 | 98.1873 | 20 |
| BINARY | 85.0095 | 84.5203 | 96.0936 | 37 | 98.6326 | 6 |
| B2 | 84.9749 | 84.4476 | 96.0513 | 71 | 98.5430 | 11 |
| B2_VALIDATION_ONLY | 84.9814 | 84.4609 | 96.0593 | 63 | 98.5641 | 12 |
| FIXED_LEGACY_VALIDATOR | 85.0277 | 84.5532 | 96.1068 | 27 | 98.6589 | 1 |
| DYNAMIC_LEGACY_VALIDATOR | 2.2149 | 0.0595 | 0.0607 | 37598 | -0.3267 | 0 |
| MLP_LEGACY_VALIDATOR | 42.6708 | 21.2558 | 27.6934 | 23789 | 36.0568 | 3533 |
| MLP_LN_LEGACY_VALIDATOR | 84.9195 | 84.3372 | 96.0038 | 202 | 98.1978 | 21 |
| BINARY_LEGACY_VALIDATOR | 85.0022 | 84.5053 | 96.0857 | 41 | 98.6220 | 8 |
| B2_LEGACY_VALIDATOR | 84.9786 | 84.4553 | 96.0566 | 67 | 98.5535 | 13 |
| B2_ORIGINAL_LEGACY | 84.9804 | 84.4589 | 96.0593 | 63 | 98.5641 | 11 |
| FIXED_BUDGET | 84.9751 | 84.4571 | 96.0593 | 58 | 98.5772 | 11 |
| DYNAMIC_BUDGET | 2.2149 | 0.0595 | 0.0607 | 37598 | -0.3267 | 11 |
| MLP_BUDGET | 19.5594 | 4.4817 | 11.3600 | 33921 | 9.3613 | 11 |
| MLP_LN_BUDGET | 84.8637 | 84.2324 | 95.9563 | 233 | 98.1161 | 11 |
| BINARY_BUDGET | 84.9469 | 84.4061 | 96.0434 | 68 | 98.5509 | 11 |
| B2_BUDGET | 84.9460 | 84.3919 | 96.0197 | 88 | 98.4982 | 11 |
| BYTETRACK | 62.9663 | 47.2484 | 62.1291 | 52 | 98.5456 | 0 |
| BYTETRACK_PAPER06 | 62.9426 | 47.2476 | 62.1736 | 52 | 98.4455 | 0 |

Pooled metrics come from one TrackEval run over all six camera sequences, not a mean of per-video metrics. Raw runs remain primary; budget-matched runs are retrospective diagnostics that alter trigger timing. Latency was measured on shared hardware, so it does not support a dedicated throughput comparison. Query diagnostics use an actual predicted-identity index but offline oracle GT-to-primary-ID alignment; this is not a deployed visual retrieval system.

## Traditional ByteTrack comparison

Pinned official source d1bf0191adff59bc8fcfeaa0b33d3d1642552a99 is executed unchanged except an old NumPy compatibility alias. At the original 0.5 preset, (0.1,0.5) contains no retained detections. At the paper-derived supplementary 0.6 threshold, 16/41/43 genuine low-score detections produce 4/10/22 accepted low-stage matches. No detection is fabricated. The cache detector floor near 0.525 truncates the original low stream. Independent per-camera Kalman tracks, cost-limited LAP and no GMT cross-camera fusion are different from the project's rejected-edge global re-solve plus learned binary revalidation. ByteTrack's lower AssA does not prove JEV architecture superiority. The supplementary 0.6 choice is source-derived and disclosed as made after some primary outcomes, not an independent primary preregistration.

## WHAT DID WE LEARN?

A native solver can change accepted assignments as an externality, but its presence does not establish learned architecture value. Ordinary normalized controls and same-solver rule controls remove the main current superiority claim. Phase VI's positive native-versus-validation-only difference is sequence-dependent and is not independently reproduced on these three new controller-heldouts. Further work should first establish feasible corrective candidates, informative action consequences and valid native support for each lifecycle question.
