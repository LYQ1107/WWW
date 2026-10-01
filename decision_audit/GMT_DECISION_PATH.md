# GMT released decision path audit

The path below was read directly from
`gtr/modeling/meta_arch/gtr_rcnn.py` at the decision-formulation branch base
commit.  The purpose is to locate the released choice rule precisely; it is
not a claim that GMT lacks a Transformer.

## Current online path

`sliding_inference_GMT()` builds a sliding history for a current view and
calls `get_asso()`.  The association model includes GMT's association
Transformer and produces association evidence for the current detections and
the retained history.  In `run_global_tracker_plus()`:

1. the final association output is activated and concatenated as
   `asso_nonk`, with shape current detections by historical observations;
2. `track_ids` on the historical observations are grouped into active unique
   Global IDs;
3. the membership matrix is
   `id_inds[history_row, id_column] = 1` when that history row has the ID;
4. GMT aggregates the evidence by ID:

   ```text
   traj_score = asso_nonk @ id_inds
   support    = id_inds.sum(dim=0)
   normalized = traj_score / max(support, eps)       # audit feature only
   ```

5. Hungarian assignment is run on the raw `traj_score`.  A proposed match is
   accepted when `traj_score[i, j] > overlap_thresh * support[j]` (or the
   configured non-multiplied threshold); otherwise the current detection gets
   a new integer track ID.  The optional memory-bank fallback handles
   unmatched observations before the normal history commit.

The decision is therefore a summed trajectory-evidence rule followed by a
fixed threshold and one-to-one assignment.  It is strictly online in the
normal path: future frames and test annotations are not inputs.

## What the decision formulation adds

The released path does not expose an explicit calibrated
`P(ID_j | current state, candidate set)`, an explicit probability for NEW,
or a decision uncertainty/verification score.  Its normalized score is a
diagnostic feature, not a probability, and each candidate is scored without
an option-set interaction layer beyond the shared sum over history.

The proposed audit keeps `asso_nonk`, the current fused ReID observation,
history statistics, and the active-ID candidate set.  It compares the GMT
rule with candidate-independent linear/MLP scorers and with a small
permutation-equivariant set choice head.  The set head emits a full softmax
over Top-K active IDs plus an explicit NEW option; calibration and
verification are measured offline before any optional dev-only choice
integration.  No COMMIT, SOFT_COMMIT, QUARANTINE, or RECOVER action is
implemented here.

## Scope and provenance

The released Stage2 checkpoint was trained on the complete VisionTrack train
set.  Consequently all results in this directory are labelled
`DECISION-FORMULATION FEASIBILITY PROTOTYPE`, not an independent base-GMT
result.  The frozen scene split is stored in
`manifests/JEV_DECISION_SPLIT.json`.  VisionTrack test images, annotations,
and metrics are out of scope and remain untouched.
