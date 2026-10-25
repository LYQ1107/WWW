# R3 preparation: spatial abstention after visual identity recovery

This isolated branch prepares one state-update repair. Frozen R2 training,
predictions, checkpoint selection and TEST jobs continue unchanged. No R3
optimizer, additional seed, TEST inference or six-GPU training is authorized
or started.

The source-level defect was established by a synthetic sequence and direct
inspection before this repair: after `VISUAL_RECOVERY_SPATIAL_CONFLICT`, the
bank retains the old valid center but resets `missed=0`. The next identical
current observation is then rejected by the geometry gate because the visual
recovery exception requires `missed>0`. This loses a correctly recovered ID.
The full fixed TRAIN VAL frequency is measured separately; this mechanism is
not assumed to explain most real errors.

R3 invalidates the conflicting old spatial state, inflates its covariance and
removes stale support points when a visual recovery is accepted. It retains
the historical appearance embedding and ID. It does not insert the rejected
new position. A later reliable current measurement may reinitialize position.
Two-view current geometry still has the unchanged pair-consistency gate.
All thresholds, neural weights, losses, candidate boxes and seed are fixed.

Before observing any R3 real-data outcome, the bounded TRAIN-only attribution
experiment is registered as follows:

- Use the archived Full selection-epoch-1 checkpoint, SHA-256
  `d4644ff95cbb7f255f70f155055fba6870863a69e7043f91b920e85ce20693dd`.
- Replay all six existing TRAIN VAL scenes 4/8/12/16/20/24 once, with identical
  frozen cache, current detections, D1024 ROI features and predicted history.
- No optimizer, teacher correction, budget reselection, new checkpoint or
  hyperparameter sweep. Run only after a diagnostic device is free.
- Compare the actual recovery/next-frame-ID events and final ID decisions with
  the immutable R2 replay using that exact checkpoint. Count regressions as well
  as corrections; report zero actual occurrences as zero.
- Official TRAIN VAL TrackEval can use only original-GT-legal scenes 4/8/12;
  16/20/24 are excluded without modifying GT. This is never a TEST result.
- The R3 source fingerprint and diagnostic authorization are separate from R2.
  R2 full-TRAIN gates remain R2 weights, not a claim of trained R3 weights.

R3-specific CPU regressions verify ID retention through the recovery and next
step, spatial abstention at the conflict frame, later current-geometry
reinitialization, and continued abstention if the next geometry is invalid.
The original package tests and three new regressions pass (203 total).
Scientific utility is not established by fixing the synthetic sequence.

The registered one-pass TRAIN replay is now complete: all six scenes, 5,860
paired times, frozen R2 selection1 weights, no optimizer and no TEST. Compared
with R2 on identical candidates, it corrects 25 and regresses 15 same-person
cross-view pairs (+10 net); corrects 100 and regresses 78 different-person pairs
(+22 net); corrects 389 and regresses 329 diagnostic temporal changes (+60 net).
All changes occur in Scene 24's clean annotation subset. The regressions are
retained, and these events are not official IDSW or TEST improvements.

On the original-GT-legal TRAIN VAL scenes 4/8/12, unchanged TrackEval gives
HOTA 74.7869011, AssA 71.7914412, IDF1 85.2940459, MOTA 90.5283908 and IDSW 319:
exactly the R2 values. Scenes 16/20/24 remain excluded without cleaning GT.
There is therefore no demonstrated formal MOT metric gain from this bounded
state repair. No long training, hyperparameter search or TEST inference follows.

The refined recovery-camera audit finds five accepted single-view recoveries:
three keep the recovered ID at the next frame, two change it, and none create
a new ID or lose the recovered camera through stale-position geometry rejection.
The R2 comparison has six recoveries, six recovered-camera changes and one
such new-ID event. These are different closed-loop event sets, so the paired
same-candidate totals above are the broader outcome comparison.

The recovery diagnostic is refined to follow the actual recovered camera,
separating its next-frame ID from another view's pre-existing split. Original
raw any-view counters remain auditable. See
`results/worldobject_v3_r3_recovered_view_refinement.json`,
`results/worldobject_v3_r3_TRAIN_only_paired_attribution.json`, and
`results/worldobject_v3_r3_TRAIN_only_result.json` for actual counts and source
bindings. The neural weights remain R2 weights; these are not a trained R3 model.
