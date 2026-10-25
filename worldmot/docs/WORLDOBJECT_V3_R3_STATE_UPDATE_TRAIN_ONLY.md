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
The original package tests are also run. Real TRAIN results remain pending
until the registered replay finishes; scientific utility is not established
by fixing the synthetic sequence.
