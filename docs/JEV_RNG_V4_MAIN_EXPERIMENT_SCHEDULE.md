# JEV v4 main-experiment schedule

Updated 2026-10-06 UTC. This schedule distinguishes the completed small-data
screening run from the final held-out tracking comparison and the later full
H=8 paper experiment.

## What has already run

The first one-seed offline screening run (`seed=20261003`) already trained:

- Learnable Threshold (`question_threshold`)
- Generic MLP (`question_conditioned_mlp`)
- Full JEV (`jev`)

Those checkpoints and the aggregated metrics are retained under the small
v6/v7 runtime artifacts and in
`reports/JEV_RNG_V4/SMALL_H8_THREE_WAY_VIDEO06_VIDEO07.json`. They are
screening evidence only: JEV matches the majority-action utility ceiling, and
the runtime parity gate is not yet passed.

## Current critical path

1. Finish the separately running corrected v4 video1 H=8 artifact. The original
   video1 small gate is complete and preserved; the corrected rebuild is the
   parity artifact. At the observed roughly 35–41 records/minute, the remaining
   data-generation time is about 3 hours from the 2026-10-06 16:30 UTC status
   window.
2. Validate the corrected artifact: 8996 records, atomic finalization, SHA,
   schema, provenance, and RNG metadata; then run the video1 old-vs-new audit.
   Expected wall time: 10–20 minutes.
3. Run GMT-OFF runtime state-feature parity and the reactivation coverage gate
   on corrected video1. Expected wall time: 30–60 minutes. Required conditions
   are exact record coverage, finite features, maximum error at most `1e-6`,
   zero OFF action mismatches, and nonzero `REACTIVATION_DECISION` coverage.
4. If those gates pass, run the three existing controller checkpoints on the
   same held-out video1: Threshold, Generic MLP, and Full JEV, with GMT OFF as
   the frozen comparison row. Do not retrain first. Estimated wall time:
   1–2 hours, depending on the runtime evaluation throughput.

Therefore, the earliest useful held-out video1 three-way tracking result is
approximately 5 hours after the 2026-10-06 16:30 UTC status window, provided
parity passes. A parity failure pauses model comparison and moves the schedule
to runtime-semantics repair.

## Full H=8 experiment

The formal three-method experiment over the complete H=8 dataset has not been
authorized. The formal 3-chunk video7 equivalence test currently has exact
semantic-key/range coverage but fails against the legacy v2 reference, so
intra-video chunking cannot be used for the canonical rebuild. The full
experiment starts only after a same-code equivalence reference passes (or a
validated single-worker fallback is explicitly accepted), followed by the
full canonical data build and audit.
