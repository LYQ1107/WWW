# Corrected v4 current-head relaunch

Date: 2026-10-06 (UTC)

## Decision

The corrected-v4 final path is pinned to source commit
`75b0aeaa7bfd3caaae942bc57b001c7c61edc8e4` (`Fix stale-bank reactivation row semantics`).
The earlier v6/v7 artifacts were started from `5cd989d` and are retained only for
diagnostic/audit purposes. They must not be used for corrected-v4 final training,
formal chunk authorization, or closed-loop tracking conclusions.

The previous small-gate three-way report is preserved as
`SMALL_H8_THREE_WAY_VIDEO06_VIDEO07_LEGACY_SMALL_GATE_SCREENING_ONLY.json`.

## Active corrected artifacts

- v6/v7 rebuild root:
  `/home/liuyeqiang/WWW_jev_rng_v4_runtime/small_h8_rng_controlled_v4_current_head_75b0aea`
- corrected canonical-feature training root:
  `/home/liuyeqiang/WWW_jev_rng_v4_runtime/small_h8_training_v4_canonical_features`
- pinned formal equivalence root:
  `/home/liuyeqiang/WWW_jev_rng_v4_runtime/formal_same_code_video7_v9_head`
- corrected video1 root (existing worker is preserved and not migrated):
  `/home/liuyeqiang/WWW_jev_rng_v4_runtime/small_h8_rng_controlled_v4_current/video_01`

All corrected v6/v7 builders and the formal workers explicitly record the pinned
source commit. The policy split and all three controllers use seed `20261003`.

## Required gates

1. Rebuild v6/v7 with the pinned source, then finalize, compact, sequence-disjoint
   split, train Threshold/MLP/JEV, calibrate, and aggregate.
2. Rebuild same-code single-worker and three-chunk video7 formal artifacts. The
   chunking authorization requires exact 3337-record semantic/canonical equality,
   exact RNG provenance, and identical canonical SHA.
3. Repeat corrected video6 runtime feature parity three times with tolerance
   `2e-5`; semantic structure, legal actions, OFF action, ordering, record count,
   missing count, and finiteness remain exact requirements.
4. Complete corrected video1 provenance, MATCH/MEMORY/REACTIVATION parity, and
   candidate parity before any final closed-loop evaluation.
5. Only after all gates pass, run GMT OFF / Threshold / MLP / JEV on the same
   mutated-state tracking sequence and publish GO/BLOCKED/NO-GO.

Full 24-video H=8 generation and intra-video chunking remain unauthorized until
the same-code formal equivalence report is PASS.

## Continuation state

The explicitly authorized canonical-feature v6/v7 pool has now been compacted
to 8501 records and trained once per method with seed `20261003`; the aggregate
report is pushed to GitHub.  The current-head v6 rebuild continues separately
for provenance/audit and must not silently replace that report.  Corrected v7
has 3337 records; corrected v6 and video1 remain live.  Formal v9 has completed
warm-up and is generating the single reference plus three chunks.  The bounded
native video1 v3 trace has been validated to contain only `video_id=1`; candidate
parity and full three-question video1 parity remain waiting on the corrected
video1 manifest.
