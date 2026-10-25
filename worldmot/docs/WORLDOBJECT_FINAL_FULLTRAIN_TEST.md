# WorldMOT World Object — full TRAIN/TEST result

## Reproducibility and scope

This is an independent branch and experiment. TRAIN contains 24 scenes and
28,754 synchronized timestamps; TEST contains 22 scenes and 29,019 timestamps.
The verified VGGT-Ω 512 checkpoint and immutable GMT candidate cache are
locked. Raw GT is never modified and no GT identity enters inference. The
full TEST cache and all three routes have 22/22 scene manifests and 29,019
timestamps.

## Official complete-video comparison

TrackEval uses the legal 14-scene TEST scope. Eight scenes are blocked by the
pre-existing raw-GT duplicate/version audit and remain coverage-only. The
following are the measured combined scores:

| route | HOTA | AssA | DetA | IDF1 | MOTA | IDSW | Frag |
|---|---:|---:|---:|---:|---:|---:|---:|
| GMT native (reference) | 66.0646 | 69.5787 | 63.0665 | 80.1121 | 72.9636 | 6,044 | 1,584 |
| GMTDet+V2 (reference) | 61.0177 | 60.2274 | 62.1099 | 74.2189 | 75.4564 | 598 | 1,289 |
| W3 persistent bank | 41.0227 | 27.2985 | 62.6754 | 38.3345 | 60.8668 | 28,800 | 1,756 |
| W4 learned update | 37.0917 | 22.2559 | 62.6995 | 34.2185 | 60.8489 | 28,834 | 1,760 |
| W5 no-geometry control | 37.0719 | 22.2392 | 62.6829 | 34.2132 | 60.8478 | 28,836 | 1,754 |

Thus W3 is the best of the three World Object routes, but it is not a GMT
improvement: it is 25.0419 HOTA points and 41.7776 IDF1 points below GMT
native. W4 is below W3, and W5 is effectively identical to W4; the learned
gate has not produced a positive TEST identity result.

The official MATLAB CVIDF1 field is unavailable for these routes in this
environment, so it is left missing rather than fabricated. A separate fixed
IoU=0.3 post-hoc cross-view diagnostic gives 82.4702 for W3/W4/W5; this is
not the official CVIDF1 protocol and must not be compared as a formal gain
over GMT's 78.0613.

## Speed definitions

The cache-build timings are real per-synchronized-pair wall-clock stages and
are reported separately from cache replay. TEST Ω model-only P50 is 5.638
pair-FPS; Ω wall P50 is 2.700 pair-FPS; total cache-step P50 is 0.668
pair-FPS. Replay-only P50 is 153.267 (W3), 112.213 (W4), and 116.901 (W5)
pair-FPS; these replay numbers do not include RGB decoding or Ω and are not
online FPS. A registered RGB→Ω→ID live timing sample (100 timestamps each from
Scenes 1, 8 and 22; 300 synchronized pairs total) includes RGB read time and
measures 0.790 pair-FPS P50 / 0.567 pair-FPS P95. This finite sample is a
speed measurement only, not a full-video accuracy result. W4/W5 retain their
replay-only speed labels because their learned-head routes were not claimed as
live measurements.

## Decision

The World Object representation is geometrically feasible (G0/G1/G2 provide
substantial diagnostic signal), but the current persistent association is not
scientifically competitive with GMT. Keep this branch as a reproducible
failure/diagnostic baseline; do not promote W4 or W5 to a mainline tracker
without a new identity-association design and a fresh registered TEST run.

Registered scientific state: `WORLD_OBJECT_FEASIBLE_BUT_BELOW_GMT`.

