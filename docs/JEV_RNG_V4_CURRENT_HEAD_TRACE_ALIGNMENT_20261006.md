# Current-head OFF trace alignment: video 06/07

This is a diagnostic checkpoint, not a final paper result. It records why the
legacy small-H8 records must not be used for the corrected v4 runtime gate.

## Finding

The legacy `video_06.jsonl` and `video_07.jsonl` traces were produced before
the current-head explicit trajectory-RNG isolation was pinned. Replaying those
actions with the current formal transformer gives a different proposal/action
map. In particular, a boundary action can create a different track, after
which later state features diverge. The large errors previously observed in
`raw_score_variance`, trajectory scores, and threshold-derived features are
therefore causal state divergence, not a GPU tolerance issue.

The legacy and current trace counts also differ:

| video | legacy events | current-head events | legacy/current action difference |
| ---: | ---: | ---: | --- |
| 06 | 5164 | 5162 | `MEMORY_DECISION` 2564 → 2562; `START_NEW` 36 → 38 |
| 07 | 3337 | 3334 | `MEMORY_DECISION` 1634 → 1631; `START_NEW` 69 → 72 |

The current-head traces are pinned to commit `98ca6caa3f7072820ab8ba1e54fd7ed3e8ff80d8`,
the Stage2 checkpoint SHA, and the explicit branch-local trajectory RNG
provenance. Their manifests are outside the Git repository because they are
large runtime artifacts; their paths and SHA256 values are recorded in
`reports/JEV_RNG_V4/CURRENT_HEAD_OFF_TRACE_ALIGNMENT_VIDEO06_VIDEO07.json`.

## Bounded replay evidence

Using 100 records and three repetitions per video:

- Video 06: no missing records, no off-action mismatch, maximum absolute
  feature error `1.52587890625e-05`; passes the explicitly tested `2e-5`
  numerical gate.
- Video 07: no missing records, no off-action mismatch, and no large semantic
  mismatch; maximum absolute error is stably `3.0517578125e-05`. This is a
  numerical-tolerance decision still to be frozen, not permission to silently
  replace `2e-5` with a looser threshold.

## Consequence and next gate

Legacy records are blocked for final training and tracking. Full current-head
records are being rebuilt from the current-head traces in new output roots, so
the old artifacts remain available for comparison. The 24-video canonical H8
rebuild remains blocked until the formal single-worker/chunk equivalence gate,
three-question parity, and candidate parity all pass.

After those gates, the sequence is:

1. compact and sequence-split the corrected records;
2. retrain Learnable Threshold, Generic MLP, and Full JEV with seed `20261003`;
3. rerun the runtime parity and mutated-state tracking pilot;
4. only then decide whether to authorize the full H8 rebuild.

