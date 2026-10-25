# WorldObject V3: joint birth repair and full-video verification

Branch: `research/worldmot-v3-birthfix-sixgpu-20261025`.
Base: `c2c9c7f7ee03b6dfa160f23b2f44d2e7339b635d`.
User instruction: attachment `c5d7a93f-e5e6-4068-83b2-45635a8dd642`.

## Status

Active revision: R2. Its 120 CPU tests and source-bound 178-timestamp real TRAIN
preflight pass. All 24 TRAIN geometry caches are now complete: 28,754 synchronized
times / 57,508 RGB images. All22 TEST geometry inputs are also complete.
Full V3 finished its selected one-epoch all24 final training and all22 TEST;
legal14 native TrackEval/MATLAB scoring is complete. Head-only continues normal
final training for its unchanged selected three-epoch budget. Full no-geometry
completed its selected two final all24 epochs and all22 TEST. Its checkpoint SHA is
`b08badfc634c9652a34366951c93207fa321deb7d1a2b240a76798f11607053f`;
see `results/worldobject_v3_no_geometry_fulltrain_gate.json`.
The completed geometry producers exited normally and were not restarted.
All four parameter-free controls have completed full TRAIN causal replay and
independent row/hash/current-candidate verification; see
`results/worldobject_v3_all_control_train_replay_verification.json`.
Full no-geometry has now completed all 22 TEST videos and official legal-14
TrackEval plus original MATLAB scoring. Full geometry also has both official
scores; Head-only TEST remains pending its final training gate.
This is an independent WorldMOT experiment; GMT is
a frozen detector/ROI provider and a separately scored reference.

## Completed official TEST results

All rows use the same original 14 legal TEST videos. The current no-geometry
model also delivered all 29,019 times of all 22 videos. Eight original-GT-blocked
videos remain excluded from official scoring.

| Method | HOTA | AssA | IDF1 | CVIDF1 | MOTA | TrackEval IDSW |
|---|---:|---:|---:|---:|---:|---:|
| GMT native reference | 66.0646 | 69.5787 | 80.1121 | 78.0613 | 72.9636 | 6,044 |
| GMTDet + V2 reference | 61.0177 | 60.2274 | 74.2189 | 72.6929 | 75.4564 | 598 |
| Full no-geometry V3 | 60.4318 | 58.2867 | 70.1622 | 67.6049 | 75.0228 | 2,130 |
| Full V3 + Ω | 56.4218 | 50.7806 | 63.9655 | 61.8413 | 74.6619 | 2,837 |

The control is below GMT and V2 in the principal identity/association metrics.
Its higher MOTA than GMT comes with fewer ID switches and nearly identical
TP/FP/FN; that numerical advantage does not imply better global identity quality.
The registered Full system falls below its capacity-matched No-Geometry control
by 4.0100 HOTA, 7.5061 AssA, 6.1967 IDF1 and 5.7636 CVIDF1 percentage points.
This experiment therefore has **no independent geometry gain** in the completed
Full-vs-NoGeometry formal comparison. The common TRAIN-only selection rule
selected different final budgets (Full 1, No-Geometry 2); this is the registered
end-to-end model comparison, not a same-weights geometry toggle. It does not
justify changing R2 or establish that every reconstruction method must fail.
Full is also below V2 and GMT. Complete metrics/checksums are in
`results/worldobject_v3_full_official_test.json` and
`results/worldobject_v3_no_geometry_official_test.json`.

Full's final checkpoint SHA-256 is
`cfda40050e89b31dd4480ca4d5010e23067a94ba18ea7d0e2425d5935d3f7fb8`.
All22 delivered candidate keys, boxes and scores match canonical detector inputs
(548,583 detections, max difference zero). Its true live timing continues as
registered; cache replay is not substituted for end-to-end FPS.

The completed No-Geometry TEST posthoc audit finds 11,662 same-person cross-view
splits out of 54,452 pairs, 123/151 detected people fragmented over IDs, and
1,515 different-person false cross-view merges. Of the latter, 1,514 reuse
existing IDs. These clean-IoU-join diagnostics are separate from official metrics;
see `docs/WORLDOBJECT_V3_R2_NO_GEOMETRY_IDENTITY_ATTRIBUTION.md`.
Lower local IDSW can coexist with stable view-specific splits and persistent
wrong ID reuse, explaining why it does not imply higher global CVIDF1.

True live GMT detection on TEST scenes 1/8/22 (100 paired times each) measures
3.3655 paired-times/s from RGB read to ID output flush, including first frames
but excluding model setup and interframe timing logs. P50/P95 are
266.279/349.607 ms. The whole sample worker including initialization, scope checks
and final hashing measures 2.5765 paired-times/s. GMT's historical 3.3313 FPS
uses all-22 full-video wall throughput, so these scopes do not establish a speed
win. The sampled device peak is 3,354,132,480 bytes including the GMT child.

`results/worldobject_v3_three_model_training_launch.json` records all 24 cache
manifest hashes, frozen scopes, actual live worker handles and UUID visibility
as observed at 2026-10-25 07:54 UTC. Model/input fingerprint remains
`6bcdcdb1a6b3683c4e406b0791416a77e5e9e64f165fbad4a61667698b64645f`.

## Repairs

- A chunked, trainable current-view pair head supplies symmetric joint-birth
  evidence to the actual constrained assignment solver. Its clean new-identity
  positives and different-identity negatives are distinct from the auxiliary
  current-view identity loss. Unmatched and ambiguous TRAIN labels are ignored.
  Candidate generation does not use GT. Logs record model/proposal retention.
- Predicted history is continuous within each scene. The 64-step boundary is
  logging/graph truncation only. Duplicate historical objects with the same TRAIN
  identity produce multi-positive historical targets instead of false births.
  Teacher history is a separate, after-forward TRAIN diagnostic.
- Point availability, alignment validity, reliability, fusion eligibility and
  world-state update eligibility are separate fields. Weak depth support cannot
  force spatial fusion or overwrite a reliable spatial state. Current detections
  remain available to the visual fallback.
- The capacity-matched appearance name is a deprecated alias of full no-geometry,
  with no duplicate training job. The no-learned-birth experiment reuses the full
  trained checkpoint with birth-pair decision evidence disabled.
- Completed cache scenes are checksum/provenance verified and reused. Partial
  scenes are preserved in a recovery directory and restarted at the scene origin.
  Atomic scene claims prevent duplicate six-worker reconstruction. A concurrent
  claim/read race was found by CPU testing and corrected.
- Native MATLAB progress is saved per job, and TrackEval output is saved before
  MATLAB. A timeout terminates only the evaluator's own process group. Native
  scoring routines and original GT are unchanged.

## Phase B evidence

`results/worldobject_v3_birthfix_cpu_test_report.json` records 105 passing CPU tests,
including the original acceptance suite. The actual Ω Python environment was used:
Torch 2.6.0+cu118, with CUDA hidden. Changed Python files also pass Pyflakes.
These are engineering tests, not real-video MOT measurements.

The first actual TRAIN preflight completed all 178 timestamps with finite losses,
nonzero assignment/birth gradients, live/replay parity and 4,998,239,744 peak
allocated GPU bytes. The identity head has 436,870 parameters; frozen Ω has
1,143,814,185. A subsequent targeted positive-birth-proposal test exposed a
NumPy integer JSON serialization error absent from that first preflight's chosen
solver proposals. It is fixed and covered by the 105th test. A second full
preflight must bind the final code fingerprint before six-card launch.
A two-timestamp TRAIN engineering run also completed actual RGB→GMT→ID without
using a detector replay. Its cold-start timing is not a formal speed result.

Final passive instrumentation records per-epoch Null/Birth/Existence prediction
counts (probability ≥0.5), low-existence counts (registered death threshold),
and separates joint solver time from identity-state updates. Live records include
head/evidence transfer, Ω model time, output serialization/flush and whole RGB-to-ID
wall time. Nested timing components must not be added twice. This instrumentation
changes no association scores or decisions and is included in the final source
fingerprint and repeated real preflight. MATLAB R2020a starts successfully and
finds the native MinCostMatching MEX; this is availability evidence, not CVIDF1.

## Frozen experiment registration

`configs/worldobject_v3_birthfix_registered.json` fixes seed 20261025, hidden size
128, the 18/6 TRAIN split, four selection epochs and the TRAIN-only extension rule
(maximum six), all-24-scene reinitialization, continuous predicted history,
joint-birth/auxiliary loss weights, geometry eligibility thresholds and candidate
limits. Full TEST contains 22 scenes; only the original 14 legally evaluable
scenes enter official aggregates. Eight blocked scenes still receive inference.
No TEST outcome may select a checkpoint or change thresholds.

Single-card engineering preflight is restricted to TRAIN scenes 1, 3, 8 and 24
(130, 16, 16 and 16 paired timestamps). Six-card execution is gated on this
preflight, checks physical GPU occupancy and binds UUIDs. Per-video latency is
reported separately from parallel aggregate throughput. Full no-geometry may
reuse immutable ROI-restored object caches; geometry methods require the new
camera-depth/persistent-alignment cache contract.

## Remaining registered measurements

Head-only final training/all22 TEST, remaining control scoring and live samples,
all retained-checkpoint TRAIN VAL attribution and the complete paired-method
posthoc audit continue. The prior V3 results are preserved and are not results
of this repaired model. No additional seed or R2 threshold change is introduced.

## Phase C and six-card launch

The final 178-timestamp TRAIN preflight passed with source fingerprint
`e0ea977f7eb6f9e9e24b75520c3dd5701c09be494913e9ea237467ea310f6ad3`.
See `results/worldobject_v3_birthfix_preflight_report.json` for raw counts, gradients,
peak allocation and launch process IDs. Earlier passing preflights remain on NAS
with the engineering reason for each source revision; only the final fingerprint
gates the long jobs. This is not multiple-seed experimentation.

Four UUID-bound workers prefer complete TRAIN scenes, one prefers TEST perception,
and one trains full no-geometry on the immutable ROI join. All five reconstruction
workers share atomic scene claims. Physical GPUs 4–9 were free at launch; other
users' devices and processes were not changed. Dataset metadata for perception
workers is physically annotation-free: TRAIN metadata SHA
`92e09f77d8d98af3ba732e75f0d2dd648313aa6e02bc058fa02a2d3d6ba97d87`, TEST
`617d76902eafe69391fccc76f581d9ba38e27bd464380681d58d14ba77efce4f`.
Original RGB directories are read through links; no images or GT are modified.
Only supervised TRAIN and later posthoc official scoring use original annotations.

Run root: `/data1/liuyeqiang/nas/A100/WorldMOT/worldobject_v3_birthfix_20261025`.
Large caches/checkpoints stay on NAS. A RUNNING job is not completed coverage.

## R2: full-TRAIN engineering audit before any TEST

The first 18-scene fit epoch exposed a blocked inference path: all 1,562 clean
new-person positive pairs reached the model, 1,389 passed the pair-head threshold,
but only three reached solver proposals. The independent singleton-birth prior
hard-pruned joint evidence. The original epoch, six-scene validation, safe
checkpoint and logs are preserved in the old run root and in
`results/worldobject_v3_pre_r2_training_audit.json`; this attempt never reached
TEST or a full-final-training gate. It is an engineering attempt, not a final
negative result for WorldMOT.

R2 keeps the registered thresholds, loss weights, seed and supervision budget.
Learned pair confidence now selects bounded birth candidates; the singleton
prior remains a soft joint-hypothesis score. Historical hypotheses can also use
the existing supervised current-pair identity evidence for dissimilar views,
while reliable conflicting geometry still vetoes a pair. The no-learned-birth
ablation disables new-pair birth evidence only and retains this historical
identity evidence, with identical frozen weights.

The beam's occupancy representation and cached tie keys are optimized separately.
Ten bounded/dense/tied/empty cases compare selected options and expansion counts
exactly with the frozen frozenset implementation. This equivalence applies to
beam representation, not to the intentional R2 birth-policy correction.
The full acceptance suite now has 120 passing CPU tests; see
`results/worldobject_v3_r2_cpu_test_report.json`.

Active run root:
`/data1/liuyeqiang/nas/A100/WorldMOT/worldobject_v3_birthfix_r2_20261025`.
Its cache directories link to the existing shared geometry producers. Thirteen
perception/coordinate/ROI source files were compared byte-for-byte; see
`results/worldobject_v3_r2_geometry_reuse.json`. Geometry inference has not changed
and is not repeated. All final neural models restart from the fixed seed in the
R2 run; pre-R2 weights are not promoted or relabelled as R2 training.

The R2 real preflight completed in 230.754 seconds with finite gradients and
4,998,239,744 peak allocated GPU bytes. Of 39 eligible same-person new-pair
diagnostic positives, 33 passed the pair threshold and 32 reached solver
proposals. These finite TRAIN engineering counts demonstrate that the blocked
path is active; they are not a TEST accuracy improvement. See
`results/worldobject_v3_r2_real_preflight_report.json`.

The dependency scheduler observes the preserved geometry workers and runs the
remaining full/head-only training, frozen-birth ablation, full-video controls,
official CPU evaluation and same-device live timing as dependencies finish.
`analysis/watch_worldobject_v3_reports.py` refreshes measured tables and runs
posthoc attribution only after all immutable predictions exist. Both scripts
have independent source hashes and never select parameters on TEST. A pending
or running row remains explicitly incomplete.

## First R2 selection epoch: mixed internal validation evidence

The 18-scene fit pass processed 22,894 timestamps, followed by all 5,860 timestamps
of the fixed six TRAIN VAL scenes. Fit assignment/joint-birth losses were
0.05533/0.00642; VAL losses were 0.20873/0.13297. Proposal recall on the
new-person diagnostic positives reached 87.02% (fit) and 81.35% (VAL), compared
with 0.19%/0.97% in the preserved pre-R2 engineering attempt.

On identical clean VAL detections, same-person cross-view sharing was
34,217/40,082 (85.37%), versus 33,961/40,082 (84.73%) before R2. Same-person
splits decreased from 6,121 to 5,865, while different-person merged pairs
increased from 2,303 to 3,232. This is mixed evidence, not a positive net-identity
or generalization conclusion. The remaining registered selection/final training
continues unchanged. These counts use diagnostic clean-label IoU matching on
TRAIN; they are not official HOTA, IDF1 or CVIDF1. Full raw details are in
`results/worldobject_v3_r2_selection_epoch1.json`.

Timing tables keep historical 22-video throughput in a separate field from the
current registered 300-pair live sample. Native GMT's historical runner delivers
final IDs at video return, so no native per-timestamp latency distribution is
inferred from that throughput. Live stage logs also expose actual detector model
and native postprocessing time. Device memory is observed through NVML, including
the GMT child context; its sampled peak is labelled separately from the exact
Torch allocation peak measured in preflight. Runtime/library versions are saved
in `results/worldobject_v3_runtime_environment.json`.

The frozen no-learned-new-birth ablation falls back to the registered heuristic
birth policy. Its contrast measures the learned new-birth path, including its
candidate selection and scoring, rather than a loss-only intervention. All
historical identity evidence and trained weights are retained. Qualitative plots
are generated only from the completed immutable posthoc audit; RGB display crops
do not change the full-image model input or scientific scores.

Completed selection epochs are also plotted in
`results/worldobject_v3_training_figures/selection_curves.png`, with source-file
hashes beside the figure. At epoch 2, VAL false-merge pairs fall from 3,232 to
2,209 while same-person split pairs rise from 5,865 to 6,293. This continuing
tradeoff is tracked without changing the registered selection rule. The resource
observer recovered from a transient NAS status-read failure, preserved its first
16,759 CSV rows, and resumed before any live timing sample; three observation
recovery tests pass. Model and perception jobs were not restarted.

The third completed no-geometry selection epoch has VAL assignment/joint-birth
losses 0.22834/0.12993, with 6,428 same-person split pairs and 3,478 false-merge
pairs. Both error counts exceed epoch 2. Selection and extension still follow
the original TRAIN-only registration; this is not a reason to tune TEST or add
seeds. The figure snapshot covers the first three completed selection epochs;
the four-epoch selection record below contains the later completed epoch.

## No-geometry selection complete; final all-24 training running

Full no-geometry completed all four registered 18/6 selection epochs. The
minimum VAL assignment-plus-joint-birth loss selects two final epochs. Epoch 4
does not improve that loss by 2% over epoch 3, so the registered extension rule
does not permit epochs 5–6. Its normalized split-plus-merge error does improve
over epoch 1; both extension conditions are required.

All rows below cover the same 5,860 synchronized times of the six TRAIN VAL
scenes. Birth-proposal recall is conditional on each epoch's causal predicted
history, whose positive sets differ. These are diagnostic counts, not official
TEST MOT metrics.

| Selection epoch | VAL assignment loss | VAL joint-birth loss | Same-person split pairs | Different-person merged pairs | Birth proposal recall |
|---|---:|---:|---:|---:|---:|
| 1 | 0.20873 | 0.13297 | 5,865 | 3,232 | 81.35% |
| 2 (selected budget) | 0.20281 | 0.13358 | 6,293 | 2,209 | 90.95% |
| 3 | 0.22834 | 0.12993 | 6,428 | 3,478 | 89.21% |
| 4 | 0.27053 | 0.12784 | 5,082 | 4,023 | 69.86% |

The first final-stage safe checkpoint was inspected on CPU: every model tensor
exactly matches a fresh seed-20261025 initialization, and the optimizer has no
accumulated state. That boundary checkpoint is preserved on NAS with SHA
`70c21b2fa504f6b0a0b72faee6495bb7ff5c4b44b167214baf2ba50b07a1b3d1`.
It is an untrained initialization, not a released inference weight. Actual
all-24 final training continues from it. Full selection history, rule evaluation
and verification are in `results/worldobject_v3_no_geometry_selection.json`.

## First completed selection epoch across the three neural methods

All three rows use the fixed six TRAIN VAL scenes, 5,860 synchronized times and
172,760 current detections after one 18-scene fit epoch. Same-person and
different-person pair denominators are 40,082 and 1,744,285 respectively.
The counts below are clean-IoU diagnostic labels, not TEST or official MOT scores.

| Method | Same-person split pairs | Different-person merged pairs | Diagnostic temporal ID changes | Model-history-absent pair splits |
|---|---:|---:|---:|---:|
| Full V3 | 4,834 | 3,056 | 4,160 | 347 |
| Full no-geometry | 5,865 | 3,232 | 2,664 | 458 |
| Head-only no-geometry | 5,790 | 2,071 | 2,878 | 96 |

Full V3 has 1,031 fewer same-person splits and 176 fewer different-person merged
pairs than full no-geometry at this epoch, but 1,496 more temporal ID changes.
This is a cross-view/temporal tradeoff, not established net identity improvement.
Full raw fit/VAL loss and diagnostic records are preserved in
`results/worldobject_v3_first_epoch_three_methods.json`. No method or threshold
is changed in response; final model selection and TEST remain as registered.

The original training key `first_appearance_split_count` means a split of a
same-person pair absent from cleanly labelled model history. It can include
repeated births after a history failure; it does not mean first physical or
first clean visibility in the video. The frozen raw key is preserved, and its
report label is corrected above. The R2 addendum separately requests both event
definitions; see `docs/WORLDOBJECT_V3_R2_SCIENTIFIC_DIAGNOSIS.md`.

The three completed parameter-free controls were also checked independently of
their completion markers. Every output row and timestamp was parsed against
physically annotation-free metadata, full file SHA values were verified, and
current candidate keys, boxes and scores matched across methods. Each delivered
24 scenes, 28,754 synchronized times and 557,002 detections. This is coverage and
input-equivalence evidence, not accuracy or live speed evidence; see
`results/worldobject_v3_control_train_replay_verification.json` and
`analysis/verify_worldobject_v3_train_replays.py`.

## Posthoc attribution acceptance

The final immutable-output audit now records false new IDs despite a unique
clean active history, serialized NULL outcomes despite a current clean person
detection, wrong-person reuse of a historical ID, and cumulative contamination
that later clean frames cannot erase. Four relative video quarters expose
accumulation over a full timeline. Exact candidate-key/bbox equality establishes
whether correctly detected observations were retained after association.

The legacy active-history reconstruction previously assumed 30 timestamp units;
its actual bank uses 12 missed processing steps. The posthoc sidecar is corrected
to that implementation, without changing any model or inference output. Legacy
does not serialize per-object NULL state, so that diagnostic's unavailable
coverage is explicit. Eight new synthetic tests pass, alongside the three
observer-recovery tests; see `results/worldobject_v3_posthoc_audit_validation.json`.
No real TEST annotations have been consumed by this new audit, which remains
gated on all eight methods' immutable complete predictions. These diagnostics
are distinct from official TrackEval and MATLAB metrics.

Live timing tables explicitly flag excluded model initialization and interframe
timing-log writes for the RGB-read-to-ID-output-flush latency measurement. A
separate complete-worker wall throughput includes model setup, scope validation,
scene resets, interframe logs and output hashing. That whole-worker duration is
not used to fabricate per-frame P50/P95. Both measurements use synchronized
two-camera times as the unit; historical full-video throughput keeps its own
scope label.
