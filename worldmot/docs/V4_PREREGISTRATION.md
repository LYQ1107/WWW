# WorldMOT V4 preregistration

Registered before reading any new V4 outcome. Base: `b448af4a7dfdf738aeab0c9dd5bc01e9a875beac`.
Branch: `research/worldmot-v4-persistent-state-20261025`. R2/R3, GT, checkpoints,
raw detection/ROI and geometry caches remain immutable. No optimizer, new seed,
new detector, full TRAIN training, or V4 TEST tracking experiment is authorized.

## Frozen inputs and finite scope

Stage A reads existing GMT Native and R2 Full/NoGeometry/HeadOnly TEST tracks.
It verifies provenance and complete videos first, reports the 14 raw-GT-legal
scenes separately from the 8 blocked scenes' clean-subset diagnostics. Matching
is one-to-one at IoU .5 with explicit ambiguous-GT exclusion. Protocol A uses
each method's matched events; Protocol B intersects detected GT events with
GMT separately for each V3 method; also report the all-method intersection.
Fixed strata: ROI cosine <=.55 / >=.90; density >=20 detections; both geometry
eligible / not. A verified camera-axis angle >=60 degrees is a separate stratum
only where poses are available; low cosine is not claimed to be a camera angle.
No-history decisions are synchronous diagnostics, not online tracking scores.

Stages B-D use all 5,860 timestamps of TRAIN scenes 4,8,12,16,20,24. Native
TrackEval uses only original legal scenes 4,8,12. The frozen **final all24 TRAIN**
Full checkpoint SHA is `cfda40050e89b31dd4480ca4d5010e23067a94ba18ea7d0e2425d5935d3f7fb8`.
These scenes were seen by final training: this is TRAIN internal development,
not a held-out generalization estimate. Full final head, hidden128, ROI1024,
birth thresholds/search/lifecycle budgets stay unchanged in every intervention.

B1: all eight G1/G2/G3 switches from the same C0 past and same current logits;
discard counterfactual states. G3 cannot change the current assignment, so also
measure post-update differences without claiming an immediate decision effect.
B2: the eight closed-loop settings, plus G4-off with G1-G3 unchanged. G4-off
is an input intervention on a geometry-trained head, not a trained NoGeometry
model. Contrast each single-factor edge and every variant against C0, reporting
both corrections and regressions. No TEST outcomes select any configuration.

## Fixed implementation sequence

C0 is bitwise-compatible legacy R2. C1 adds CV prediction/Joseph Kalman update.
C2 adds uncertainty-weighted temporal geometry, never distance-only hard veto.
C3 adds separate identity/spatial safe updates. C4 adds conservative multi-view
measurement and identity reliability coordination. C5 adds causal Ω scheduling.
G1 and birth logic stay legacy in C0-C4. G4 shape/weights stay unchanged.

Time is VisionTrack `frame_id` ticks, not assumed seconds. No verified FPS is
provided by COCO metadata. Velocity is world-units/tick. Ω world scale is
relative; no meter claim. CV acceleration spectral density defaults to .01
times the initial measurement variance per tick^3, initial velocity variance
equals position variance per tick^2; these are engineering priors, not calibrated
metric noise. Measurement covariance is support scatter inflated by point/depth
confidence, alignment reliability and visibility, never divided by point count.
Temporal spatial bonus/penalty is bounded to +/- .5 with stale reliability decay
30 ticks; trusted-current quality .6, identity recovery uses original cosine .95
and learned margin2, plus unambiguous historical competition. Long-term EMA .15
times confidence; short-term per-camera EMA .5. No search over these priors.
Covariance eigenvalues must be finite and positive; invalid input abstains.
Multi-view covariance includes disagreement and is not halved for correlated
views; no repeated independent updates from the same Ω call.

C5 requests at scene start; new/ambiguous identities, competing objects, a
reappearance with insufficient visual evidence, untrusted spatial state, or
30 ticks without trusted geometry. Visual competition margin .05 cosine;
no current Ω values/GT are inspected before request. On skip, observations have
no current geometry. Cache request count is a simulation, not a LIVE speed.
Live reactivation uses an actual last trusted keyframe; failed alignment
abstains and cannot reuse a stale Sim(3) as success.

## Acceptance and stop boundary

Run all C0-C5 even if one fails, but do not add parameter variants based on
outcomes. Report sync splits/merges, temporal changes, history rejection/reuse,
false birth, NULL, mixed-ID contamination, fragmentation, one-to-one retention,
gap recovery, native TRAIN VAL HOTA/AssA/IDF1, and stage P50/P95/runtime.
Verify legacy exactness against the unmodified R2 core with the same frozen
inputs. CPU tests cover all 20 requested failure classes with numerical checks.
Finite live paired replay is allowed only after implementation tests and cache
validation. It must share device, precision, source and latency scope, include
decode/wait/output, and distinguish initialization, model-only and RGB-to-ID.

Advance recommendation requires attributable TRAIN gains, no hidden merge
regressions, passing engineering tests and a frozen policy. A negative result
is retained. New full training/TEST requires explicit subsequent user approval.

Within the finite C1-C4 set, a candidate must strictly improve all three native
TRAIN HOTA/AssA/IDF1, have positive net temporal corrections, and nonnegative
same-person/different-person net sync corrections. Choose among qualifying
TRAIN candidates by AssA, then IDF1, then HOTA, then lower complexity. This is
TRAIN development selection, not a new TEST selection. C5 LIVE scheduling is
gated specifically on a qualifying C4; unsuccessful C4 still retains its finite
cache/CPU diagnostics. Do not invent new variants after observing outcomes.

Finite LIVE engineering timing, if a free authorized V100 is available: first
30 timestamps of TRAIN scenes 4/8/12, identical physical device and precision for
C0-C4, and C5 only after its state-quality gate. Each variant gets the same five
TRAIN-scene4 warmup timestamps, all tracking state discarded, before measured
scene origins. Include read, IPC/wait, model, alignment, state and flushed ID
output. Report initialization/warmup separately. This is not full-video FPS.

The historical t704->t705 ID158->162 example belongs to retained Full selection
epoch1 (`d4644ff95cbb7f255f70f155055fba6870863a69e7043f91b920e85ce20693dd`),
not necessarily the final all24 Full checkpoint. Any exact-number reproduction
must use that separately identified retained checkpoint; it cannot be silently
substituted into the primary C0-C5 comparison.
