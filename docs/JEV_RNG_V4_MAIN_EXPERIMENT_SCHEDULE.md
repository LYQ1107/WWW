# JEV v4 main-experiment schedule

Updated 2026-10-06 UTC. This schedule distinguishes the completed small-data
screening run from the final held-out tracking comparison and the later full
H=8 paper experiment.

## Current gate status (2026-10-06 UTC)

The corrected-v4 final tracking comparison is still fail-closed. The old
small-gate checkpoints and report remain available, but are
`LEGACY_SMALL_GATE_SCREENING_ONLY` and are not valid for corrected-v4 final
tracking.

- Corrected video1 rebuild: still running; do not stop or migrate it.
- Corrected video6/video7 compact, split, and one-seed retraining: pending
  corrected video6 finalization.
- Same-code formal video7 single-vs-three-chunk equivalence: running; chunking
  remains unauthorized until the all-record verifier passes.
- Three-repeat corrected feature-parity tolerance probe: pending corrected
  video6 finalization; the candidate protocol is `2e-5`, not a silent
  `1e-4` relaxation.
- Bounded video1 reactivation candidate parity: preliminary FAIL (14 native
  events vs 5 replay events, with candidate/state-score divergence); this is a
  runtime-semantic gate, not a tracking result. It must be rerun after the
  corrected video1 artifact is complete.

## What has already run

The first one-seed offline screening run (`seed=20261003`) already trained:

- Learnable Threshold (`question_threshold`)
- Generic MLP (`question_conditioned_mlp`)
- Full JEV (`jev`)

Those checkpoints and the aggregated metrics are retained under the small
v6/v7 runtime artifacts and in
`reports/JEV_RNG_V4/SMALL_H8_THREE_WAY_VIDEO06_VIDEO07.json`. They are
screening evidence only: JEV matches the majority-action utility ceiling, and
the runtime parity gate is not yet passed. They must not be used for the
corrected-v4 final tracking comparison.

## Current critical path

1. Finish the separately running corrected v4 video1 H=8 artifact. The original
   video1 small gate is complete and preserved; the corrected rebuild is the
   parity artifact. At the observed roughly 35 records/minute, the remaining
   data-generation time is about 3 hours from the 2026-10-06 17:18 UTC status
   window.
2. Validate the corrected artifact: 8996 records, atomic finalization, SHA,
   schema, provenance, and RNG metadata; then run the video1 old-vs-new audit.
   Expected wall time: 10–20 minutes.
3. Run corrected v6/v7 compact, sequence-disjoint split, and one-seed
   retraining in parallel as soon as video6 finalizes. This produces the only
   checkpoints eligible for corrected-v4 tracking.
4. Complete the three-repeat feature-parity stability probe and freeze the
   numerical tolerance. Structural fields remain exact; the current candidate
   is `2e-5` if all repetitions pass.
5. Run GMT-OFF runtime state-feature parity and the reactivation candidate
   gate on corrected video1. Required conditions are exact per-question
   coverage, finite features, the frozen numeric tolerance, zero OFF action
   mismatches, and exact candidate coverage/order/score checks.
6. Only if all gates pass, run corrected-v4 Threshold, Generic MLP, and Full
   JEV on the same held-out video1, with GMT OFF as the frozen comparison row.
   Estimated wall time: 1–2 hours, depending on runtime throughput.

Therefore, the earliest useful held-out video1 three-way tracking result is
approximately 4–5 hours after the 2026-10-06 17:18 UTC status window, provided
all semantic gates pass. A parity failure pauses model comparison and moves the
schedule to runtime-semantics repair.

## Full H=8 experiment

The formal three-method experiment over the complete H=8 dataset has not been
authorized. The formal 3-chunk video7 equivalence test currently has exact
semantic-key/range coverage but fails against the legacy v2 reference, so
intra-video chunking cannot be used for the canonical rebuild. The full
experiment starts only after a same-code equivalence reference passes (or a
validated single-worker fallback is explicitly accepted), followed by the
full canonical data build and audit.
