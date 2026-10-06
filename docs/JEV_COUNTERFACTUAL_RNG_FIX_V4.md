# JEV counterfactual RNG isolation and REASSOCIATE semantics — v4

Status: the v4 implementation gates pass, the completed small-gate artifacts
are reproducible, and the first closed-loop pilot is `PILOT_FAIL`. The 24-video
canonical H=8 rebuild is therefore not authorized yet.

The active small-gate worker for video1 remains running on GPU6. It must not be
stopped or migrated: its throughput is about 35 records/min, comparable to
video6/video7 per record; it is slower only because it has 8,996 decisions.

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
- The runtime pilot recomputes controller features from the mutated state and
  passes one proposal through feature extraction, resolution, and commit.

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
| Learnable Threshold | 0.895657 | 0.780015 | 0.009979 | 0.687612 | 36.58994965 |
| Generic MLP | 0.894163 | 0.962432 | 0.008617 | 1.072714 | 36.63555383 |
| Full JEV | 0.894633 | 0.999806 | 0.008971 | 1.152154 | 36.64465531 |
| Majority-action reference | — | 0.999806 | — | — | 36.64465531 |

JEV technically beats both learned controls, but its validation utility is
exactly the majority-action reference and its margin over MLP is only about
`0.0091`. This is not strong evidence for a closed-loop tracking claim.
The full report is `reports/JEV_RNG_V4/SMALL_H8_THREE_WAY_VIDEO06_VIDEO07.json`.

## Early closed-loop pilot

The pilot uses held-out video1 (`00002garden`) and the same frozen detector/ReID
cache and formal GMT backend for GMT OFF, Threshold, MLP, and JEV. It is
explicitly screening-only: the tracking trace is the preserved legacy trace,
not a new v4 H=8 record set. There are currently zero new v4 records available
for video1 while its builder is still running. The runtime state-feature parity
gate is blocked (`expected_records=0`, `compared_records=0`, and 24 OFF action
mismatches), so these metrics must not be treated as final paper results.

| Method | HOTA | AssA | IDF1 | MOTA | IDSW | Frag | ΔAssA |
|---|---:|---:|---:|---:|---:|---:|---:|
| GMT OFF | 86.5218 | 85.1464 | 94.6425 | 89.6246 | 280 | 6 | 0 |
| Learnable Threshold | 88.3191 | 88.7342 | 97.0634 | 94.1297 | 82 | 6 | +3.5878 |
| Generic MLP | 59.0078 | 39.6298 | 48.9800 | 52.9238 | 1893 | 6 | −45.5166 |
| Full JEV | 86.1798 | 84.4984 | 93.0509 | 95.9044 | 4 | 6 | −0.6480 |

The Threshold screening run is positive, but it does not satisfy the required
“MLP/JEV positive” condition. The Generic MLP catastrophically collapses
association. JEV sharply reduces ID switches but lowers HOTA/AssA/IDF1. The
pilot result is therefore `PILOT_FAIL_NO_GO_FOR_CANONICAL_REBUILD`.
See `reports/JEV_RNG_V4/SMALL_H8_RUNTIME_TRACKING.json` and
`reports/JEV_RNG_V4/FINAL_GO_NO_GO.json`.

## Current decision

`canonical_full_h8_rebuild_authorized = false`.

Keep the current video1 small-gate builder running. Do not launch the 24-video
canonical H=8 rebuild yet. The immediate investigations are:

1. verify controller label/action semantics against the runtime commit path;
2. explain the MLP `REASSOCIATE`/`SKIP_MEMORY` collapse and JEV association
   regression;
3. finish video1, then rerun a parity-valid pilot using v4 records;
4. evaluate and validate deterministic intra-video chunking before any
   canonical rebuild authorization.

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
an orchestration test only; a bounded formal-GMT equivalence run remains a
mandatory precondition for canonical authorization.

This design targets the future video15/video23/video24 tail bottlenecks and is
not applied to the active video1 small gate.
