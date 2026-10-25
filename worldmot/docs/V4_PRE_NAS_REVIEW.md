# V4 offline verification before NAS recovery

This review adds no real-data result and selects no parameter from TEST. The
immutable checkpoint/cache/trajectory checks remain pending until NAS recovery.

The source-bound CPU receipt currently contains 250 tests, zero failures and
zero skips. The original 200 package tests remain in the suite. In addition to
the numeric CV/Kalman, identity, visibility and matching assertions, the online
frontend now has a complete synthetic request/skip/skip/request sequence:

* Ω is called only at timestamps 0 and 3.
* Timestamps 1 and 2 contain current visual detections and predicted spatial
  state, without old depth being presented as a current measurement.
* At timestamp 3 the fresh reconstruction has a different local gauge. The
  aligner actually estimates the static-correspondence Sim(3), including an
  independent test with scale 2 and translation 10 in relative world units.
* The recovered persistent center and the two cameras' shared Global ID remain
  unchanged within numerical tolerance; the trusted geometry time advances to 3.
* A separate regression repairs the diagnostic status of an intentional Ω skip:
  it is `OMEGA_SKIPPED_NO_CURRENT_GEOMETRY`, not `WORLD_ALIGNMENT_FAILED`.
* A corrupt prior covariance is detected, spatial evidence abstains, and a
  trusted current observation reinitializes a positive covariance without
  changing the current person's ID or suppressing the detector box.

These are synthetic engineering assertions, not reconstruction quality or person
association measurements on VisionTrack. The 3-tick refresh in the unit fixture
does not change the registered real policy's 30-tick setting.

## Interpretation boundaries fixed before outcomes

1. C1 changes the state estimator to a six-dimensional CV/Kalman estimator. Its
   initialization and propagated uncertainty differ from R2's static center
   accumulator. Any C1 gain must be attributed to this estimator change; it
   cannot automatically prove that accurate person velocity was learned.
2. The A audit uses a common maximum-cardinality then maximum-IoU matcher at .5.
   Earlier R2 clean-subset diagnostics used greedy one-to-one matching. A's
   counts may therefore differ from the old +2,208/+618 diagnostic. Native
   TrackEval/MATLAB and original GT are unchanged.
3. A Full-vs-GMT correction is an observed difference between two systems.
   Geometry-eligibility, appearance and camera-angle strata do not isolate Ω's
   causal contribution. B's fixed-weight interventions provide the separate
   TRAIN mechanism evidence.
4. G4-off is an input/branch intervention on the frozen geometry-trained head.
   It is not a newly trained No-Geometry head and cannot alone establish the
   benefit of learning geometry from scratch. The separately trained R2
   No-Geometry reference remains explicitly identified.
5. C5's cache replay simulates requests and state consumption. Its cached world
   coordinates were produced by the original continuous aligner, so it does
   not validate a sparse LIVE reactivation's alignment quality or its speed.
   The actual LIVE path uses fresh reconstruction and historical anchors.
6. The final all24 TRAIN head has seen the fixed six TRAIN VAL scenes. These
   experiments are internal development and causal diagnostics, not independent
   generalization estimates. No new V4 official TEST tracking is authorized.
7. Noise inflation and confidence gates are registered engineering priors,
   not claims of calibrated metric uncertainty. World units remain relative
   and time remains frame-id ticks unless a verified frame rate is supplied.

The exact retained epoch1 t704/t705 case, native TRAIN scores, actual correction
and regression counts, and LIVE latency remain **unmeasured** at this stage.
Passing synthetic tests does not satisfy those outstanding requirements.
