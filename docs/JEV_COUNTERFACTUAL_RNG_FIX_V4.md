# JEV counterfactual RNG isolation and REASSOCIATE semantics — v4

Status: the v4 implementation gates pass, the first closed-loop pilot exposed a
train/runtime state mismatch, and the 24-video canonical H=8 rebuild is not
authorized yet. The original video1 small gate is complete; its corrected v4
rebuild is now running for final parity.

The original video1 small-gate worker was allowed to finish naturally on GPU6
and was not stopped or migrated. It produced 8,996 atomic records at about 35
records/min. A separate corrected v4 video1 worker now uses the same free GPU6
only after that small gate completed; its output is the parity artifact, not a
replacement for the historical small-gate record set.

## Scope and frozen inputs

The frozen GMT checkpoint, H=8 horizon, utility definition, state schema,
detector/ReID cache, legal actions, and formal association transformer are
unchanged. The v4 change is limited to counterfactual provenance and action
evaluation semantics. The complete VISION_test GMT baseline was not rerun.

The old Full H=8 runtime at `/home/liuyeqiang/WWW_jev_full_h8_runtime` was
paused with SIGTERM and preserved. Its records are
`RNG_UNCONTROLLED_SCREENING_ONLY`, `PRE_RNG_CONTROL`, and
`NOT_FINAL_CANONICAL_H8`; see `reports/JEV_RNG_V4/OLD_H8_PAUSE_SNAPSHOT.json`.

## v4 implementation

- `TrajectoryRandom` preserves Python `random.sample`/`random.choices` output
  while making the trajectory RNG explicit and recording a mapping digest.
- Formal adapter calls restore a fresh RNG from
  `state.trajectory_rng_state`; missing state fails closed.
- Each video starts from master seed `20261006 + video_id`.
- `MutableGMTState.clone()` deep-copies tracker, memory, history, and RNG
  provenance.
- `propose()` is observational. A committed `step()` advances the branch RNG
  exactly once for the initial transformer proposal.
- `resolve_actions()` and `step()` accept a precomputed proposal.
  `REASSOCIATE` reuses its score tensor and performs only constrained Hungarian;
  it makes no second transformer call.
- Counterfactual records now seed the native first-frame GMT view and build
  state features from the current mutable OFF state through the same v2 state
  schema used by runtime inference. They no longer copy the legacy trace's
  state feature vector as training input.
- The runtime pilot recomputes controller features from the mutated state and
  passes one proposal through feature extraction, resolution, and commit.
- The runtime pilot now also evaluates stale online memory through the formal
  GMT association proposal and exposes `REACTIVATION_OLD`/`START_NEW` as a
  typed `REACTIVATION_DECISION`, with reactivation assignments committed through
  mutable state only. GT and future records are not used to trigger it.

## Completed deterministic gates

All five synthetic/source-level gates are `PASS`:

- `RNG_BRANCH_ISOLATION.json`
- `ACTION_ORDER_INVARIANCE.json`
- `REASSOCIATE_PROPOSAL_REUSE.json`
- `REPEATABILITY.json`
- `WORKER_INDEPENDENCE.json`

For the completed video7 artifact, the independent repeat is exact:
3,337/3,337 canonical records match, raw record SHA-256 is identical, best
actions have zero mismatches, and state features, target probabilities, and
utilities have zero maximum delta. See
`REPEATABILITY_ARTIFACTS_VIDEO07.json`.

The video6/video7 old-vs-new audit compares all 8,501 records and reports
97.4121% best-action agreement overall (video6 99.9613%, video7 93.4672%). It
is a screening audit, not proof of strict old-trace parity, because the legacy
trace has no complete RNG state/call provenance. See
`OLD_VS_RNG_FIXED_DATASET_AUDIT_VIDEO06_VIDEO07.json`.

## First single-seed offline comparison

The completed v6/v7 compact dataset uses the sequence-disjoint split and the
locked seed `20261003`. Training is one run per method, 20 epochs, batch 128,
AdamW, learning rate `1e-3`, with matched trainable capacity. The validation
sequence is video6; video7 is the policy-training sequence. Video1 remains
held out for tracking.

| Method | Val NLL | Accuracy | Brier | ECE | Validation utility |
|---|---:|---:|---:|---:|---:|
| Learnable Threshold | 0.895657 | 0.780015 | 0.009979 | 0.343806 | 36.58994965 |
| Generic MLP | 0.894163 | 0.962432 | 0.008617 | 0.543081 | 36.63555383 |
| Full JEV | 0.894633 | 0.999806 | 0.008971 | 0.576077 | 36.64465531 |
| Majority-action reference | — | 0.999806 | — | — | 36.64465531 |

The ECE implementation now passes hard probability/calibration invariants. JEV
technically beats both learned controls, but its validation utility is exactly
the majority-action reference and its margin over MLP is only about `0.0091`.
The tie/margin audit reports a 51.84% best-action tie rate on validation, so
this is not evidence for a meaningful learned-policy advantage or a closed-loop
tracking claim.
The full report is `reports/JEV_RNG_V4/SMALL_H8_THREE_WAY_VIDEO06_VIDEO07.json`.

## Train/runtime mismatch diagnosis and correction

The first small-gate records were not safe training data: they copied the
legacy trace feature vector and replayed the first frame from an empty state.
On a video6 runtime replay through frame 20, this produced 163 compared
features with maximum absolute error `31.580963` and mean absolute error
`0.098956`. The largest errors were `raw_score_variance`,
`score_over_threshold`, `raw_traj_score`, and `score_minus_threshold`.

Commit `0c31b72` adds production seed initialization, native first-frame view
ordering, and `canonical_state_feature_for_event`; the runtime pilot can now
use the same formal GPU device for parity. A bounded formal probe on 120
corrected video6 records reduced the maximum feature error to `1.53e-5`, mean
error to `9.74e-9`, with zero OFF action mismatches. The diagnostic is recorded
in `reports/JEV_RNG_V4/TRAIN_RUNTIME_FEATURE_MISMATCH_DIAGNOSTIC.json`.

An earlier corrected full video6/video7 rebuild is isolated under
`/home/liuyeqiang/WWW_jev_rng_v4_runtime/small_h8_rng_controlled_v3_canonical_features`
and remains separate from the completed small-gate artifacts. No new video6,
video7, or repeat7 generation is required for the current gate.

## Early closed-loop pilot

The pilot uses held-out video1 (`00002garden`) and the same frozen detector/ReID
cache and formal GMT backend for GMT OFF, Threshold, MLP, and JEV. It is
explicitly screening-only: the tracking trace is the preserved legacy trace,
not a new v4 H=8 record set. The bounded legacy smoke exercised four
`REACTIVATION_DECISION` records (`REACTIVATE_OLD=0`, `START_NEW=63`) but failed
legacy runtime feature parity (`expected_records=8790`, `compared_records=2297`,
`missing_records=6493`, maximum error `1357.4833984375`). These metrics must not
be treated as final paper results; corrected v4 parity is still pending.

| Method | HOTA | AssA | IDF1 | MOTA | IDSW | Frag | ΔAssA |
|---|---:|---:|---:|---:|---:|---:|---:|
| GMT OFF | 86.5218 | 85.1464 | 94.6425 | 89.6246 | 280 | 6 | 0 |
| Learnable Threshold | 88.3191 | 88.7342 | 97.0634 | 94.1297 | 82 | 6 | +3.5878 |
| Generic MLP | 59.0078 | 39.6298 | 48.9800 | 52.9238 | 1893 | 6 | −45.5166 |
| Full JEV | 86.1798 | 84.4984 | 93.0509 | 95.9044 | 4 | 6 | −0.6480 |

These numbers are retained only as legacy-trace screening. Because the v4
record parity gate has not passed and this smoke is not the corrected v4
artifact, the tracking result is
`RUNTIME_PARITY_BLOCKED_INCONCLUSIVE_FOR_MODEL_SELECTION`; it is not model
FAIL evidence.
See `reports/JEV_RNG_V4/SMALL_H8_RUNTIME_TRACKING.json` and
`reports/JEV_RNG_V4/FINAL_GO_NO_GO.json`.

## Current decision

`canonical_full_h8_rebuild_authorized = false`.

Keep the corrected v4 video1 rebuild running. Do not launch the 24-video
canonical H=8 rebuild yet. The immediate gates are:

1. finish the corrected v4 video1 artifact without stopping or migrating it;
2. run provenance/SHA plus the
   old-vs-new audit;
3. pass v4 runtime feature parity and REACTIVATION coverage, then rerun the
   true closed-loop pilot with the already-trained checkpoints;
4. complete the formal-GMT video7 chunk equivalence gate before any canonical
   rebuild authorization.

## Required canonical-only chunking design

The small gate intentionally does not use temporary intra-video sharding. Before
the 24-video rebuild, the repository must provide a deterministic chunk mode:

1. Build a stable plan from ordered `(video, frame, view)` decision units, with
   chunk boundaries only between complete decision units.
2. Run one quick GMT-OFF warm-up in production order and checkpoint the exact
   `MutableGMTState`, including tracker/memory/history containers and
   `trajectory_rng_state`, at every chunk start.
3. Run chunks on separate GPUs. Each worker loads only its start snapshot,
   replays its non-overlapping decision range, and extends replay through the
   H=8 look-ahead needed by its last decision. It emits only decisions owned by
   its range.
4. Merge by the stable semantic key
   `(video_id, frame, view, question_type, detection_index)` and reject
   overlaps, gaps, duplicate keys, wrong snapshot provenance, or wrong source
   hashes.
5. Run a bounded single-worker build and the chunked build on the same video,
   then require exact per-record equality: raw canonical record fields, best
   action, target probabilities, utility outcomes, state feature vectors, and
   RNG provenance. A failed equivalence gate blocks canonical use.

The implementation is now present in `jev_intra_video_chunking.py`,
`plan_jev_intra_video_chunks.py`, `warmup_jev_intra_video_chunks.py`,
`run_jev_intra_video_chunk.py`, and
`verify_jev_intra_video_chunk_equivalence.py`. A bounded CPU cosine-contract
probe on the first 120 video7 trace lines split the data into three chunks and
matched all 120/120 records exactly with identical canonical SHA-256. This is
an orchestration test only. The formal-GMT 3-chunk run completed with
`3337/3337` semantic keys and valid contiguous ranges, but failed exact
equivalence against the existing `small_h8_rng_controlled_v2/video_07`
reference: raw canonical matches were `0/3337`, state features differed on
`3337/3337`; the shared partition/range/RNG provenance checks pass, but the
legacy source commit and feature schema differ. This is a
data-generation/chunk-reference failure, not evidence that Threshold, MLP, or
JEV failed. The result is recorded in
`reports/JEV_RNG_V4/FORMAL_GMT_INTRA_VIDEO_CHUNK_EQUIVALENCE_VIDEO07.json`,
and intra-video chunking remains unauthorized. A same-code corrected
single-worker reference is required before reconsidering chunk authorization.

This design targets the future video15/video23/video24 tail bottlenecks and is
not applied to the active video1 small gate.
