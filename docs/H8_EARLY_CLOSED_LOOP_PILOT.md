# Full H=8 Early Closed-Loop Pilot

**Date:** 2026-10-06
**Classification:** `SCREENING_ONLY / NOT_FOR_FINAL_SELECTION / NOT_FOR_PAPER_RESULT`

## Executive conclusion

The read-only H=8 snapshot is structurally healthy. The first tracking replay
did expose a real engineering bug in the pilot harness: it reconstructed the
controller feature vector with the wrong GMT window semantics. That result is
preserved in Git history but is superseded by the canonical-feature parity
replay below.

- The snapshot passed schema, checkpoint, state, utility, and finiteness checks.
- All three models learned an offline signal in the narrow sense that their
  validation NLL/accuracy are non-random, but the majority policy remains
  slightly better in validation utility. Full JEV is not yet selected over
  Generic MLP by the offline gate.
- After replacing the incorrect hand-built feature vector with the exact
  formal `state.feature_vector`, Full JEV and Learnable Threshold show
  plausible positive association signal on held-out `00021gate`. Generic MLP
  regresses but no longer catastrophically collapses.
- The corrected screening verdict is
  **`PILOT_GO_FOR_FULL_H8_CONTINUATION`**. This is not a paper result or a
  final controller-selection GO. The formal Full H=8 builders were not stopped
  or modified.

## Provenance and isolation

The pilot used a read-only snapshot of the formal runtime at
`2026-10-06T05:26:30Z`. It read complete JSONL records from complete shards and
from the byte boundary of running temporary shards. No formal source file was
edited, and an incomplete final JSONL line would have been dropped only from
the pilot copy.

- Snapshot records: **110,364**
- Source shards: **24**
- Invalid records: **0**
- Horizon: **H=8**
- Checkpoint SHA256:
  `cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8`
- Formal builders modified: **false**
- Policy train: **100,957** records
- Policy validation: **5,457** records from `00002garden`
- Tracking test: **3,950** records from held-out `00021gate`
- Train/validation/tracking sequence intersections: **empty**

The policy split is by whole sequence/video group; it is not a random record
split. The tracking replay used the two views of `00021gate`, 523 frames per
view, and did not use the policy train or validation sequence.

## 1. Data audit

| Check | Result |
| --- | ---: |
| `MATCH_DECISION` | 55,581 |
| `MEMORY_DECISION` | 54,612 |
| `REACTIVATION_DECISION` | 171 |
| State finite rate | 1.000000 |
| Zero sample-weight rate | 0.000000 |
| Best-action tie rate | 46.2107% (51,000) |
| Utility tie rate | 44.5100% (49,123) |
| Utility min / max | -57.5 / 674.5 |
| Utility mean / stdev | 146.5312 / 147.5945 |
| Suspicious degeneration flags | none |

Best-action counts were `ACCEPT_CURRENT=48,787`, `REASSOCIATE=5,977`,
`START_NEW=6,755`, `WRITE_MEMORY=50,236`, `SKIP_MEMORY=51,449`, and
`REACTIVATE_OLD=170`. The data does not show the requested >95% single-action
or >90% utility-tie degeneration, but the high tie rates mean that raw
best-action accuracy is not a sufficient selection criterion.

## 2. Offline three-way screening

All methods used the same pilot train/validation split, seed `20261003`,
10 epochs, batch size 256, AdamW, learning rate `0.001`, and CPU execution.

| Method | Val NLL | Accuracy | Brier | ECE | Validation utility |
| --- | ---: | ---: | ---: | ---: | ---: |
| Learnable Threshold | 0.953859 | 0.976910 | 0.062647 | 0.795173 | 41.076507 |
| Generic MLP | 0.899831 | 0.989738 | 0.012881 | 1.013032 | 41.079531 |
| Full JEV | 0.900762 | 0.978010 | 0.012485 | 1.014317 | 41.076599 |
| Majority-by-question | — | 0.994869 | — | — | **41.081501** |
| Uniform random legal action | — | 0.681571 | — | — | 40.930013 |

The offline gate is false: Full JEV is below Generic MLP in validation utility
and only narrowly above Threshold. The existing ECE implementation reports
values above 1 in this weighted screening output; those values are retained as
diagnostics and are not used to claim calibration quality.

The training curves are stored in `PILOT_OFFLINE_THREE_WAY.json` under each
method's `training.loss_curve` field.

## 3. Held-out closed-loop tracking replay

This is a mutable-state replay using the formal GMT association transformer
and typed controller commits. It reuses frozen detector/ReID perception
payloads from the formal cache; it does **not** rerun live detector inference
and it is not the official full `VISION_test` evaluation. The same payloads,
GT, checkpoint, evaluation mapping, and TrackEval procedure were used for all
four rows. The evaluation was strict-online and used raw GT; this sequence had
zero duplicate GT rows.

The first run used a hand-built feature vector with `window_length=523` and
missing canonical counters. It produced the previously committed catastrophic
MLP/JEV numbers, but those numbers are **invalid for scientific interpretation**
because the controller did not receive the same feature schema used in
training. The corrected run uses the exact online-only `state.feature_vector`
from the formal H=8 trace wherever the held-out record exists: 3,950 canonical
decision records, with 48 explicitly counted fallbacks. Its result is the
authoritative pilot output.

### Absolute metrics

| Method | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Frag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GMT OFF | 85.6188 | 86.1685 | 85.0745 | 94.9401 | 89.1013 | 67 | 14 |
| Learnable Threshold | 85.8429 | 86.1816 | 85.5070 | 95.1357 | 89.6272 | 56 | 13 |
| Generic MLP | 74.5421 | 84.3058 | 65.9099 | 85.8959 | 86.9981 | 111 | 13 |
| Full JEV | **86.0505** | **86.1991** | **85.9036** | **95.4779** | **90.2008** | **44** | 13 |

### Delta versus the same GMT OFF replay

| Method | ΔHOTA | ΔAssA | ΔIDF1 | ΔMOTA | ΔIDSW | ΔFrag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Learnable Threshold | +0.2240 | +0.4325 | +0.1956 | +0.5258 | -11 | -1 |
| Generic MLP | -11.0767 | -19.1646 | -9.0442 | -2.1033 | +44 | -1 |
| Full JEV | **+0.4317** | **+0.8291** | **+0.5378** | **+1.0994** | **-23** | -1 |

The corrected controller action-rate diagnostic is:

| Method | ACCEPT | REASSOCIATE | START_NEW | WRITE_MEMORY | SKIP_MEMORY | wrong-commit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GMT OFF | 48.849% | 0.000% | 1.151% | 50.000% | 0.000% | 9.241% |
| Threshold | 48.874% | 0.400% | 0.725% | 50.000% | 0.000% | 9.215% |
| Generic MLP | 45.973% | 3.102% | 0.925% | 39.670% | 10.330% | 11.620% |
| Full JEV | 48.949% | 0.375% | 0.675% | 1.426% | 48.574% | 5.544% |

The corrected replay recorded one missing action-outcome lookup for Threshold
and one for Full JEV. These were retained as diagnostics and treated as zero
contamination contribution; they did not change the controller decision or the
GMT state. The corrected Full JEV signal is still screening-only and must be
validated on the completed H=8 data and additional seeds.

## 4. Direct answers to the pilot questions

1. **Does the current H=8 data look reasonable?** Yes at the schema/state/
   utility level. It is not obviously degenerate, although ties are high.
2. **Can the three controllers learn?** Yes, all three learn an offline
   predictive signal. Offline utility is close to the majority-policy ceiling,
   so this does not select Full JEV by itself.
3. **Is there positive tracking signal?** Yes after feature parity: Full JEV
   improves HOTA by `+0.4317`, AssA by `+0.8291`, and reduces IDSW by 23 on
   this held-out sequence. Threshold is also slightly positive; MLP is a
   moderate regression, not a catastrophic collapse.
4. **Should we continue generating the remaining roughly one million records?**
   Yes, continue the already-running formal Full H=8 builders. The corrected
   pilot no longer shows a catastrophic runtime failure, but do not freeze the
   final method from one sequence/one seed; finish Full H=8, then run the
   locked multi-seed and official evaluation protocol.

## 5. Immediate follow-up plan

1. The 50-context checkpoint round-trip parity test now passes for all three
   methods with zero mismatches and zero probability error; retain it as a
   regression gate.
2. Let the formal H=8 builders finish and preserve their resumable shards.
3. Re-run the three methods on the completed canonical H=8 policy split and
   then run the locked `20261004`/`20261005` robustness seeds.
4. Only after those gates, run the official tracking comparison; do not use
   this one-sequence pilot as a paper table.

## Reproducibility artifacts

The repository copy contains the small JSON summaries and the source scripts.
The full JSONL snapshot, policy dataset, detector cache, prediction arrays,
and model checkpoints remain in the runtime directory and are intentionally
not committed to GitHub.

- `PILOT_DATA_AUDIT.json`
- `PILOT_OFFLINE_THREE_WAY.json`
- `PILOT_TRACKING_RAW.json`
- `PILOT_TRACKING_THREE_WAY.json`
- `PILOT_RUNTIME_PARITY.json`
- `PILOT_POLICY_SPLIT.json`
- `PILOT_SNAPSHOT_MANIFEST.json`
