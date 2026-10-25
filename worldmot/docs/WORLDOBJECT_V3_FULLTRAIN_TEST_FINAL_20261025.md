# WorldObject V3: joint birth repair and full-video verification

Branch: `research/worldmot-v3-birthfix-sixgpu-20261025`.
Base: `c2c9c7f7ee03b6dfa160f23b2f44d2e7339b635d`.
User instruction: attachment `c5d7a93f-e5e6-4068-83b2-45635a8dd642`.

## Status

Active revision: R2. Its 120 CPU tests and source-bound 178-timestamp real TRAIN
preflight pass. Fresh no-geometry selection training is running, alongside five
unchanged geometry producers from the preserved pre-R2 checkout. No TEST has started and no
new accuracy result is claimed. This is an independent WorldMOT experiment; GMT is
a frozen detector/ROI provider and a separately scored reference.

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

## Results awaiting execution

Training coverage, full TEST coverage, official metric comparison, birth/error
attribution, and live runtime breakdown will be added as real jobs finish.
The prior V3 results are preserved and are not results of this repaired model.

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
