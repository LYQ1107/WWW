# JEV Runtime State Contract v3

Status: implementation contract for the `jev/runtime-state-contract-v3-20261006`
branch.  This document is a gate specification, not a paper result.

## Scope

The controller must see the same online state semantics in all four places:

1. formal GMT trace collection;
2. counterfactual policy-record construction;
3. closed-loop pilot replay;
4. future official GMT inference.

The detector and ReID payloads are frozen evidence.  The tracker state,
association history, memory and typed action commits are mutable and branch
local.  No future annotation or evaluator value is allowed in runtime state.

## Single source of truth

`gtr/modeling/jev_state.py` is the only feature encoder.  Runtime code calls:

- `build_state_values(...)` for named online values;
- `build_state_features(...)` for the 64-dimensional tensor;
- `encode_state(...)` for stable schema packing.

The shared helpers also define candidate entropy, the legacy GMT acceptance
threshold, bounded association-history counts, memory observation counts and
the GMT association-window length.  Pilot and replay code must not duplicate
these transformations.

The schema keeps bounded evidence bounded (`accept_score`,
`reassociate_score`, flags and normalized counts), while the accumulated
trajectory tail remains raw evidence (`raw_traj_score`, mean, `log1p`,
score-minus-threshold and score-over-threshold).  This preserves the existing
checkpoint/training semantics.

## Association RNG provenance

The released GMT transformer currently assigns trajectory embedding slots with
the process-global Python RNG on every association call.  This is part of the
association implementation, not controller state.  A historical trace is
therefore replayable only if it records the RNG seed/state or the exact slot
mapping used for each window.  The existing H=8 trace does not record either.

The v3 pilot sets `random.seed(20261003)` to make a new replay internally
repeatable, but that cannot reconstruct the old process's unrecorded RNG
stream.  The bounded probe consequently reports a parity failure (maximum
feature error `2.4690799713134766` versus the `1e-6` gate) even though the OFF
actions still match on that probe.  This is a provenance failure, not evidence
that the mutable-state contract is correct.

Until a trace with explicit association-RNG provenance is generated, runtime
mode is fail-closed: it runs OFF only, writes the parity and divergence
reports, and does not execute learned controllers or publish tracking metrics.
`trace_debug` remains diagnostic-only and cannot bypass this runtime gate.

## GMT threshold contract

`cfg.VIDEO_TEST.OVERLAP_THRESH` is the base threshold and is passed unchanged
to the formal replay engine.  With `NOT_MULT_THRESH=False`, the legacy GMT
decision compares a trajectory score against:

```text
base_threshold * max(1, track_length)
```

With `NOT_MULT_THRESH=True`, it compares against the base threshold.  The
feature named `accept_threshold` always contains the base threshold; it is not
silently replaced by the scaled value.

## Mutable replay order

The replay follows `GTRRCNN.sliding_inference_GMT`:

- select the first-frame view with the most detections; ties choose the
  highest view index;
- seed that view with track IDs `1..N`;
- process the remaining first-frame views in ascending view order;
- process later frames in ascending view order;
- trim historical slices before each current view to the same
  `TEST_LEN`-bounded prefix used by native GMT;
- use `T` on the first-frame secondary view and `T // view_num` on later
  multi-view decisions.

For each current slice, `resolve_actions()` is non-mutating.  The final
existing identity is resolved before memory questions are generated:

- `START_NEW` receives no same-frame memory question;
- `REASSOCIATE` memory uses the final committed identity;
- only `step()` applies track, memory and history mutations.

## Parity gates

The runtime pilot is valid only if the OFF branch passes all of these gates:

- every saved pilot record is reconstructed;
- every reconstructed state feature is finite;
- runtime and saved trace vectors agree per dimension with max absolute error
  no greater than `1e-6`;
- reconstructed OFF actions equal the saved formal OFF actions;
- OFF tracking metrics reproduce the known pilot reference:
  `HOTA 85.6188`, `AssA 85.0745`, `IDF1 94.9401`, `MOTA 89.1013`,
  `IDSW 67`, `Frag 14`.

`trace_debug` may substitute the saved vector for controller checkpoint
debugging, but its tracking result is labelled `TRACE_DEBUG_ONLY`.  It is not
evidence for a closed-loop claim.  Runtime mode always recomputes the vector
from the mutated state and fails closed if the OFF gate is not satisfied.  A
short smoke pass is not a full-sequence parity pass.

## Required artifacts

The pilot writes the following repository artifacts:

- `reports/JEV_RUNTIME_STATE_V3/STATE_FEATURE_PARITY.json`;
- `reports/JEV_RUNTIME_STATE_V3/CHECKPOINT_RUNTIME_PARITY.json`;
- `reports/JEV_RUNTIME_STATE_V3/TRACE_DEBUG_TRACKING.json`;
- `reports/JEV_RUNTIME_STATE_V3/RUNTIME_STATE_TRACKING.json`;
- `reports/JEV_RUNTIME_STATE_V3/DIVERGENCE_ANALYSIS.json`.

All are screening diagnostics.  They do not replace the frozen canonical GMT
baseline or constitute final paper metrics.  The existing Full H=8 builders
on `jev/reviewer-proof-v2` remain independent and must not be stopped or
modified by this branch.
