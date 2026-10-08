# P0.5 preregistered attribution protocol

This protocol is frozen before viewing new heldout outcomes. TRAIN controller-heldouts are **09,
10, 11**, the three lowest video numbers without documented controller selection use. Videos
01–08, 23 and 24 have earlier research use; 08 was the original early pilot. Training remains
07; calibration remains 06. The GMT backbone may have seen TRAIN: this is not an official test.

## Historical B2 attribution and supervision limitation

To attribute the *existing* B2 result, fit a generic MLP (64→153→153→3, 33,969 parameters) and
state-conditioned scalar threshold (64→154→154→1 plus base threshold, 34,036 parameters) on
exactly B2's 4,303 historical records (07 train 1,703; 06 calibration 2,600), soft utilities,
sample weights, 20 epochs, AdamW lr .001, batch 128, seed 20261003, fixed final checkpoint.
Calibration uses 06 only. B2 has 34,080 parameters. No heldout selection or retry after collapse.
These labels inherit H8 prefix resets/frozen OFF continuation and are **not native causal truth**.
Therefore this comparison can attribute historical B2 deployment behavior, but cannot establish
the value of native long-horizon supervision. Binary JEV is the frozen Phase VI binary-trained
control with the same historical inputs/update budget; its retained binary utility is an explicit
supervision/action ablation, not an identical-target architecture comparison.

## Policies and native transition

Primary: GMT OFF, fixed .1 threshold, dynamic threshold, generic MLP, Binary JEV with explicit
three-action adapter, frozen B2. All receive identical canonical 64-dimensional state and the
same legal first-round actions. Rules see the canonical raw current score (56), clipped alternate
score (1), and legal mask. Fixed/dynamic accept when current score exceeds their threshold;
otherwise request REASSOCIATE if alternate exceeds the same threshold, else START_NEW.
Dynamic soft logits use a hierarchical accept / rejected-current-and-acceptable-alternate /
new partition. Binary JEV accepts according to its binary head; after rejection the common .1
alternate gate selects REASSOCIATE or START_NEW. This adapter is disclosed separately from
the unmodified binary controller. All three actions are reachable without GT.

Every REASSOCIATE calls the **actual** `GTRRCNN._apply_jev_match_decisions`, full score matrix,
native banned-edge mask, constrained Hungarian and native binary second-round feature generation.
ACCEPT rows are allowed to move. Own-validator primary uses each policy's binary restriction.
No-GT inference; MEMORY/REACT remain GMT. Perception and candidate-generation/score functions
are frozen. Paired starts have identical scores/candidates; complete mutated-state futures are
recomputed and may diverge endogenously. No caching factual future association scores.

Controls: B2 VALIDATION_ONLY (native second validation, original pairs; no global reassignment),
B2 global+legacy second validator, B2 original pairs+legacy validator, and common-legacy-validator
versions of rule/dynamic/MLP/Binary. Together these identify first policy vs validation vs global
effects and a global×validation interaction conditional on B2. No-validation is not unconditional
acceptance: it uses GMT's unchanged score/track-length threshold.

## Paired intervention and budget diagnostic

Capture the first 12 payloads per heldout where B2 has a legal REASSOCIATE proposal, without
GT filtering. Compare all policies and factorial controls from each exact cloned prefix/RNG.
Evaluate immediate committed IDs and live H8/H16/H32 continuation on a bounded first four
captures per sequence. Ties, unknown GT and right censoring remain visible. Only these native
records can enter later MiniSet gates; historical labels stay separate.

Budget diagnostic targets the **observed B2 native trigger count** per sequence using an online
quota schedule based only on frozen sequence length and current event index. Within the current
payload rank legal rows by the policy's own REASSOCIATE propensity; force/suppress REASSOCIATE
to follow cumulative target. No future score or GT is read. Report achieved budget and any deficit;
call exact-budget comparisons valid only when achieved counts match. Include B2 under the same
quota schedule to distinguish timing effects. This retrospective-count diagnostic is not a primary
deployable policy and never replaces raw runs. Results cannot select heldouts or hyperparameters.

## Traditional tracking and low detections

Read ECCV 2022 ByteTrack and pinned official tracker `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`.
Its high/low sequential IoU/Kalman association is different from this rejected-edge global re-solve.
Official-style parameters are fixed before outcomes: high >.5, low .1–.5, match_thresh .8,
track_buffer 30, frame_rate 30; separate camera trackers, no invented cross-camera fusion.
The frozen heldout cache contains zero detections in that low interval: label the low stage
NOT_EXERCISED and any official run HIGH_ONLY_AVAILABLE_INPUT, never a complete ByteTrack
low-stage validation. Do not regenerate perception or fabricate discarded detections.

## Metrics, statistics and decision gates

Report every sequence plus **true pooled TrackEval**, not a mean labelled pooled: HOTA, AssA,
IDF1, IDSW, MOTA, Frag where evaluator supports it. Log rejected edges, REASSOCIATE rows/payloads,
pair changes, ACCEPT moved, second-validation acceptance, correction/regression, solver/controller
latency, total wall time. Offline identity error duration uses a stable identity mapping (prefix
mapping for paired forks; sequence-level identity matching for full videos), never frame-wise remapping.
Unknown/missed observations are separate; report frame units and censoring, not invented seconds.

Architecture-support GO requires all contracts, nontrivial causal outcomes, full three-sequence
positive HOTA and AssA vs both dynamic and MLP, pooled ≥.1 HOTA and ≥.2 AssA vs both, and
consistent direction under an actually equal trigger budget. Three sequences support descriptive
replication only; do not call a positive mean statistically significant. Traditional-explanation
diagnostic: if same-solver fixed/dynamic/MLP reaches ≥80% of B2's positive pooled gain, stop
expanding the current state-only MATCH architecture. If B2 gain is nonpositive, the gain fraction
is undefined and architecture extension is also unsupported. Candidate conditioning is only a
conditional next hypothesis after native causal/candidate gates, not an automatic success.

No new candidate architecture training before P0.5. No data gate is relaxed after seeing outcomes.
WHAT DID WE LEARN? Fairness requires the same solver, legal actions and validator contract;
old binary-only baseline results cannot answer this attribution question.
