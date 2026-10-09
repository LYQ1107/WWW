# Phase XIII final research report

All primary models below use verified true Stage1 VFCE1024, full natural TRAIN12/13/14/16, common legal candidates/native executor, and fresh20,000-update checkpoints. Three seeds are evaluated on all complete development17/18/19 videos. Scores are actual pooled TrackEval over six cameras, followed by seed means. These development scenes were used historically and were exposed during inherited Stage1 pretraining.

| Method | HOTA | AssA | IDF1 | IDSW | MOTA | Frag |
|---|---:|---:|---:|---:|---:|---:|
| full | 75.056 | 76.717 | 91.137 | 118.667 | 89.741 | 307.333 |
| motip | 74.527 | 75.671 | 89.979 | 145.000 | 89.676 | 307.667 |
| camel | 74.757 | 76.202 | 90.692 | 130.333 | 89.708 | 307.333 |
| set_transformer | 74.836 | 76.318 | 90.894 | 158.333 | 89.634 | 308.333 |
| no_question_reader | 75.088 | 76.833 | 91.317 | 120.333 | 89.737 | 307.000 |
| fixed_question | 75.637 | 77.907 | 92.431 | 105.000 | 89.780 | 307.000 |
| no_option_reader | 75.162 | 76.939 | 91.889 | 199.333 | 89.543 | 309.000 |
| no_gating | 74.269 | 75.169 | 90.062 | 180.333 | 89.596 | 306.333 |
| no_cross_camera | 72.225 | 71.104 | 86.333 | 308.667 | 89.256 | 308.667 |
| no_long_term | 62.494 | 53.435 | 72.304 | 272.667 | 89.340 | 302.000 |
| no_competition | 74.210 | 74.997 | 89.667 | 139.667 | 89.690 | 306.667 |
| no_typed_head | 75.091 | 76.815 | 91.706 | 157.333 | 89.644 | 308.000 |
| B1 cosine | 69.085 | 65.052 | 80.021 | 82.000 | 89.818 | 303.000 |
| Original GMT separate full system | 78.110 | 80.821 | 94.814 | 204.000 | 91.547 | 258.000 |

Full improves raw HOTA by5.970 over cosine, but its mean IDSW violates the frozen1.25×cosine limit. Its HOTA advantage over the strongest ordinary control is only0.219.
Fixed Question reaches HOTA75.637 and No Question Reader reaches75.088, versus Full75.056. The current dynamic Question Reader benefit is unsupported. Removing cross-camera or long-term evidence harms tracking, but that evidence is also available to the ordinary controls; these ablations do not establish a unique structured-decision advantage.

## Native official cross-camera metric engine

Unchanged official evaluateTracking/CLEAR_MOT_HUN/IDmeasures and repository MEX ran in MATLAB R2020a on every frozen primary case. The VisionTrack benchmark parameter skips MOT16-only class/visibility cleaning: the original converter uses unknown auxiliary values of -1, which MOT16 defaults would incorrectly treat as low visibility. The failed-format diagnostic is retained in MATLAB_FORMAT_COMPATIBILITY_AUDIT.json. Sequential camera blocks use max(GT,pred)+1; CVMA uses interleaved frame*n_views+view_index, two-decimal geometry and the native end-frame clipping. Counts are pooled across scenes. Canonical outputs are secondary and separately recorded.

| Method, 20k raw primary | CVIDF1 mean ± seed std | CVMA mean ± seed std |
|---|---:|---:|
| full | 90.249 ± 0.765 | 87.202 ± 0.735 |
| motip | 88.837 ± 2.863 | 86.404 ± 1.153 |
| camel | 89.724 ± 0.479 | 87.632 ± 0.439 |
| set_transformer | 89.977 ± 0.746 | 86.051 ± 1.307 |
| no_question_reader | 90.703 ± 1.742 | 87.937 ± 0.591 |
| fixed_question | 91.889 ± 0.477 | 87.888 ± 0.542 |
| no_option_reader | 91.526 ± 1.432 | 86.980 ± 1.575 |
| no_gating | 88.675 ± 1.963 | 86.187 ± 1.344 |
| no_cross_camera | 61.588 ± 2.843 | 40.967 ± 2.626 |
| no_long_term | 70.721 ± 2.251 | 82.687 ± 2.394 |
| no_competition | 87.926 ± 1.579 | 84.066 ± 1.084 |
| no_typed_head | 90.565 ± 1.598 | 86.420 ± 2.080 |
| cosine | 64.843 ± 0.000 | 46.450 ± 0.000 |
| Original GMT separate Stage2 full system | 94.814 | 91.416 |

## Bounded own-state round

The following comparison uses the same completed eligible seeds before and after the extra4k updates. Wrong-anchor observations count actual errors relative to each identity first GT anchor; they must be read together with births/fragments and full tracking metrics. Prefix error duration is censored and is not a counterfactual propagation estimate.

| Method | Eligible completed seeds | 20k HOTA paired | 24k HOTA | 20k IDSW paired | 24k IDSW | TRAIN wrong-anchor before | after | TRAIN extra births before | after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 3 | 75.056 | 75.635 | 118.667 | 143.667 | 1768.000 | 796.333 | 35.667 | 65.000 |
| motip | 3 | 74.527 | 74.698 | 145.000 | 190.333 | 1205.333 | 1031.333 | 41.000 | 80.667 |
| camel | 3 | 74.757 | 74.549 | 130.333 | 145.000 | 1301.333 | 965.333 | 36.000 | 30.000 |
| set_transformer | 3 | 74.836 | 74.519 | 158.333 | 152.667 | 1716.000 | 988.667 | 37.333 | 52.000 |

Observed prefix fragmentation tradeoffs:

- full seed20261008: wrong-anchor observations 2698 → 990, extra birth fragments 40 → 132.
- full seed20261010: wrong-anchor observations 1221 → 902, extra birth fragments 30 → 32.
- motip seed20261008: wrong-anchor observations 1110 → 828, extra birth fragments 36 → 37.
- motip seed20261009: wrong-anchor observations 1526 → 879, extra birth fragments 51 → 171.
- set_transformer seed20261009: wrong-anchor observations 1052 → 938, extra birth fragments 35 → 92.

## Actual live efficiency

All trials use one empty V100, real image loading, frozen Stage1 detector/VFCE and native committed histories. First256 scene frames have two camera payloads each; camera FPS is shown separately. Frozen thresholds are Stage2 p95≤10ms and full two-camera scene FPS≥25. An initial live threshold mismatch (0.525 versus frozen0.55) was repaired; cached/live boxes, VFCE and actual IDs are verified in each result. The failed first attempt and the Full256-frame exact parity audit remain available.

| Method | Full scene FPS | Camera payload FPS | Stage2 p50 ms | Stage2 p95 ms | Full peak VRAM MiB | Budget |
|---|---:|---:|---:|---:|---:|---|
| cosine | 4.603 | 9.205 | 7.616 | 9.466 | 642.5 | FAIL |
| full | 4.674 | 9.349 | 12.185 | 13.563 | 658.0 | FAIL |
| motip | 4.250 | 8.501 | 14.376 | 20.099 | 659.1 | FAIL |
| camel | 4.332 | 8.663 | 14.452 | 16.108 | 658.3 | FAIL |
| set_transformer | 4.323 | 8.647 | 14.542 | 16.062 | 657.5 | FAIL |
| no_question_reader | 4.744 | 9.488 | 10.444 | 12.373 | 659.5 | FAIL |
| fixed_question | 4.701 | 9.401 | 11.755 | 13.623 | 658.8 | FAIL |
| no_option_reader | 4.376 | 8.753 | 11.666 | 15.452 | 658.0 | FAIL |
| no_gating | 4.352 | 8.703 | 12.926 | 14.109 | 658.4 | FAIL |
| no_cross_camera | 4.298 | 8.597 | 14.177 | 19.307 | 658.8 | FAIL |
| no_long_term | 4.654 | 9.308 | 12.167 | 13.745 | 658.8 | FAIL |
| no_competition | 4.382 | 8.764 | 12.804 | 14.426 | 658.9 | FAIL |
| no_typed_head | 4.345 | 8.691 | 13.737 | 15.701 | 659.3 | FAIL |

The parameter-free cosine controller also misses the full-scene FPS budget with this common Stage1 frontend/native pipeline. Full separately exceeds the Stage2 latency limit. These single256-frame trials do not establish a statistical speed ranking between methods.

Original GMT uses Stage2-trained perception/RPCE and is a separate strong full-system comparator. MOTIP-style/CAMEL-style are controlled adapters, not complete official framework reproductions.

Native state parity exposed a restored-ID recycling bug before primary training. Old data/models/results remain archived. The correction adds a recovered ID back to the real possible-ID set; the real GTA-throw lifecycle test now recovers the same ID twice and verifies full/resumed state through106 frames. All primary models use the regenerated v3 corpus. A dormant-state issue in the Set control was also repaired before training; every shared-state/ordinary decoder block has nonzero supervised gradients.

Tiny/Pilot use frozen examples and budgets; formal primary checkpoint is LAST at20k. All GT labels are offline and uncertain/contaminated candidate histories remain UNKNOWN. REACT has only9 natural clean TRAIN positives and MEMORY has0 WRITE/KEEP labels, so learned MATCH+REACT and three-lifecycle experiments are NOT_RUN. Common cosine recovery/native WRITE remain explicit fallbacks. The NoTyped result cannot identify typed lifecycle benefit under MATCH-only supervision.

On-policy round status: COMPLETE. Its frozen bound is first256 frames of each of four TRAIN videos, one4k update round per eligible main model/seed, with its own actual mutated histories. Total24k outcomes are kept separate from20k; before/after observed wrong-ID counts and censored duration are in ON_POLICY_TRAINING.json.

No-calibration runs use the exact same Full checkpoints with T=1 on all three development videos. Actual raw-prediction hashes and metric deltas are in ARCHITECTURE_ABLATION.json; positive uniform temperature should preserve a unique maximum-sum assignment, and calibration benefit is restricted to certified MATCH probabilities.

Actual isolated live image/detector/VFCE/native measurements, executed matrix-multiply cost, latency p50/p95 and full VRAM are in EFFICIENCY.json. Cache-backed validation timing is never called full FPS. Native MATLAB R2020a was installed from the user-supplied ISO, and untouched official evaluator/MEX passed analytic perfect/miss/false-positive/ID-split fixtures. Actual native official CVIDF1/CVMA over all frozen raw and canonical cases are in OFFICIAL_MATLAB_CROSSVIEW.json. The separately labelled Python TrackEval adaptation remains available for comparison.

| Gate | Result |
|---|---|
| G0 | PASS_WITH_PRETRAIN_EXPOSURE_LIMIT |
| G1 | PASS |
| G2 | PASS_MATCH_ONLY |
| G3 | PASS |
| G4 | PASS |
| G5 | FAIL |
| G6 | PASS |
| G7 | NO_GO |
| G8 | FAIL |
| G9 | NO_GO |

Frozen exploratory independent-value gate: NO_GO. These three preexposed scenes do not establish population-level significance. See exact per-seed paired deltas and structural ablation means in FINAL_GO_NO_GO.json.

Heldout20/21/22 remain sealed: inherited Stage1 already saw every TRAIN video, so an independent full-system heldout claim is invalid. Full24 and official TEST remain unauthorized. A genuinely unseen feature-level evaluation requires an explicitly authorized perception retraining/split redesign before reopening heldout.

Large datasets/checkpoints/predictions/segmented native logs remain on the server. Git contains only source, protocols, compact results/plots and SHA references. All PhaseV–XII scientific assets and failures remain intact. RUN_ENVIRONMENT.json plus each run binding records actual source/data/weights/config/seed provenance.
