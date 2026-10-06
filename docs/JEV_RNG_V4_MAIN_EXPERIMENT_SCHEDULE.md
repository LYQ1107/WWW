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
- The first `small_h8_rng_controlled_v3_canonical_features` v6/v7 artifacts
  are retained but their manifests report source commit `ec40eff`, before the
  production-seed/state corrections. They are not current-HEAD final data.
- A current-HEAD v6/v7 rebuild is running under
  `/home/liuyeqiang/WWW_jev_rng_v4_runtime/small_h8_rng_controlled_v4_current`;
  only these new manifests can authorize corrected-v4 controller training.
- Same-code formal video7 single-vs-three-chunk equivalence: the earlier
  source-mixed run is not authoritative; a current-HEAD single, warm-up, and
  three-chunk run is now running. Chunking remains unauthorized until its
  all-record verifier passes.
- The three-repeat video7 preliminary parity probe is reproducible but FAILS:
  `1699/1705` compared records, 10 OFF-action mismatches, and maximum feature
  error `1685.5164794921875`. This is a state/semantics failure, not a reason
  to relax `2e-5` to `1e-4`; the required corrected video6 probe is still
  pending.
- Same-native-cache bounded reactivation candidate parity is FAIL (`14` native
  events vs `1` replay event). IDs/order on the overlapping event agree, but
  event coverage and one score disagree. This is a runtime-semantic gate, not
  a tracking result, and must be rerun on the final corrected video1 source.

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
3. Finish the current-HEAD v6/v7 rebuild, then compact, make the
   sequence-disjoint split, and retrain Threshold/MLP/JEV once with
   `seed=20261003`. This produces the only checkpoints eligible for
   corrected-v4 tracking.
4. Complete the three-repeat corrected video6 feature-parity stability probe
   and freeze the numerical tolerance. Structural fields remain exact; the
   candidate is `2e-5` only if all repetitions pass.
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
authorized. The first formal 3-chunk video7 equivalence test had exact
semantic-key/range coverage but compared different source revisions. The
current-HEAD same-code replacement is running; only a PASS from that artifact
can authorize intra-video chunking for the canonical rebuild. The full
experiment starts only after that gate and the runtime semantic gates pass,
followed by the full canonical data build and audit.
