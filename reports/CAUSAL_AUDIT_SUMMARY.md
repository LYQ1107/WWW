# GMT causal audit summary

This report records the fixed three-scene causal audit in the isolated
`jev-gmt-causal-prototype` worktree.  The released Stage2 checkpoint and the
inference recipe were kept fixed.  The audit seed was `20260930`, with
`TEST_LEN=40`, the memory bank enabled, and `BANK_SIZE=10`.  GT was read only
by the audit sidecar; it was not placed in tracker tensors or association
features.  Event schedules were frozen from C0 before intervention outcomes
were inspected.

## Runs

| run | purpose | runtime (s) | manifest events | executed/sham | tracking metrics (HOTA / IDF1 / AssA / MOTA) |
|---|---|---:|---:|---:|---|
| C0_baseline_log | baseline event discovery | 1904 | — | — | 83.068 / 82.634 / 81.924 / 79.622 |
| C1_correction | oracle correction | 1025 | 71 | 70 / 1 conflict | 83.182 / 82.802 / 82.343 / 79.298 |
| C1_sham | correction sham | 1892 | 71 | 71 | 83.068 / 82.634 / 81.924 / 79.622 |
| C2_injection | controlled wrong-ID injection | 1040 | 939 | 535 / 395 conflicts / 9 inactive | 80.167 / 79.296 / 76.749 / 77.950 |
| C2_injection_sham | injection sham | 1863 | 939 | 939 | 83.068 / 82.634 / 81.924 / 79.622 |

The first correction attempt and first injection attempt are retained under
`causal/runs/*_failed_*`; neither failed artifact was used for the reported
effects.  The correction retry only skipped a target when an earlier online
intervention made that ID occupied or inactive in the current view.  The
injection retry applied the same deterministic rule.  The frozen manifests
were not regenerated.

## Primary paired effects

The endpoint is future identity error relative to the frozen correct target
ID; the event frame itself is excluded.  Interventions are paired with their
same-event sham and use bootstrap seed `20260930`.

| intervention | +1 | +2 | +5 | +10 | +20 |
|---|---:|---:|---:|---:|---:|
| C1 correction − sham | +0.083 [0.008, 0.167] | +0.073 [-0.009, 0.155] | +0.061 [-0.026, 0.149] | +0.111 [0.009, 0.222] | +0.116 [0.027, 0.214] |
| C2 injection − sham | +0.385 [0.344, 0.426] | +0.377 [0.335, 0.420] | +0.376 [0.334, 0.419] | +0.362 [0.319, 0.405] | +0.356 [0.313, 0.399] |

Positive values mean more future error.  Thus the injection has a strong,
consistent damaging effect, while the oracle correction does not reduce error
and is positive on the pooled estimate at every horizon.

## Gate

The predeclared gate is **CONDITIONAL GO**: C2 meets the expected positive
direction at lag 5/10/20, but C1 does not meet the expected negative direction.
No horizon satisfies the strong-both-interventions requirement, and only one
intervention supports causal propagation.

Per the project protocol, the study stops here.  JEV-GMT design, offline head
training, online integration, and any formal GMT training were not started.

Detailed evidence is in [CAUSAL_EVIDENCE_MATRIX.md](../causal/CAUSAL_EVIDENCE_MATRIX.md),
[C1_ORACLE_CORRECTION.md](../causal/reports/C1_ORACLE_CORRECTION.md), and
[C2_ERROR_INJECTION.md](../causal/reports/C2_ERROR_INJECTION.md).
