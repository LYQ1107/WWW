# Candidate identity mapping audit — video01

Status: `PASS_WITH_LIMITATIONS`. This is a read-only feasibility audit, not a
training result, paper result, or Full-H8 authorization.

## What was checked

The audit uses the immutable segmented native trace, the VisionTrack training
annotations, and the frozen perception-cache index. It reconstructs the frame-0
seed view and maps prior accepted native track observations to GT instances by
box IoU. Candidate IDs are then compared with the current detection's GT
identity. It never copies `ACCEPT_CURRENT`, `REASSOCIATE`, or `START_NEW`
action labels into candidate labels.

The exact inputs and SHA-256 values are recorded in
`reports/JEV_RNG_V4/CANDIDATE_IDENTITY_MAPPING_AUDIT_VIDEO01_20261007.json`.

## Results

| Question | Events | Candidate rows | Rows with prior identity | New/unresolved rows | Current-GT candidate recall | Top-1 rate |
|---|---:|---:|---:|---:|---:|---:|
| MATCH | 4,524 | 56,048 | 13,748 | 42,300 | 99.794%* | 99.680%* |
| REACTIVATION | 174 | 559 | 476 | 83 | 0.000%* | 0.000%* |

`*` Recall and top-1 are conditioned on current detections that could be
matched to a GT box at IoU >= 0.5: 4,377/4,524 MATCH events and 131/174
REACTIVATION events. The MATCH rank distribution when the current identity is
present is rank 1: 4,363, rank 2: 2, rank 3: 3; mean rank is 1.0018.

The 42,300 unresolved MATCH rows are not automatically errors: they are
candidate IDs with no prior GT-backed observation, typically newly created
track IDs. Their identity cannot be inferred from the numeric ID namespace.
They must receive an independently computed branch-return utility or remain
unlabelled.

REACTIVATION is a hard negative for the proposed candidate-conditioned route:
none of the 131 mapped current detections has its current GT identity in the
native reactivation candidate set. Therefore this audit does not support
training or claiming a candidate-conditioned REACTIVATION improvement.

## Consequence for the next experiment

Only the MATCH candidate path is allowed to proceed, and only after building
per-candidate long-horizon returns by isolated frozen-evidence rollouts. The
next experiment must keep candidate sets legal and variable-length, avoid
absolute track IDs as features/classes, and use the same sequence split and
held-out tracking test. This audit alone does not authorize training, runtime
closed-loop evaluation, or Full H8.
