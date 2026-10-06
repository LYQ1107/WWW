# Full H=8 Early Closed-Loop Pilot

**Date:** 2026-10-06
**Classification:** `SCREENING_ONLY / NOT_FOR_FINAL_SELECTION / NOT_FOR_PAPER_RESULT`

## Executive conclusion

The read-only H=8 snapshot is structurally healthy, but the current controller
runtime integration is not ready for a full tracking comparison.

- The snapshot passed schema, checkpoint, state, utility, and finiteness checks.
- All three models learned an offline signal in the narrow sense that their
  validation NLL/accuracy are non-random, but none beats the majority policy in
  validation utility. Full JEV does not beat Generic MLP.
- The held-out closed-loop replay is decisive for this screening run:
  Learnable Threshold is slightly positive, while Generic MLP and Full JEV
  catastrophically damage association. Full JEV is therefore **not a GO**.
- The strict screening verdict is **`PILOT_FAIL` for the current learned-policy
  runtime path**. The formal Full H=8 builders were not stopped or modified;
  however, the remaining data must not be treated as sufficient justification
  for final controller selection until the offline/runtime semantic mismatch is
  diagnosed.

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

This is a real mutable-state replay using the formal GMT association transformer
and typed controller commits. It reuses frozen detector/ReID perception
payloads from the formal cache; it does **not** rerun live detector inference
and it is not the official full `VISION_test` evaluation. The same payloads,
GT, checkpoint, evaluation mapping, and TrackEval procedure were used for all
four rows. The evaluation was strict-online and used raw GT; this sequence had
zero duplicate GT rows.

### Absolute metrics

| Method | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Frag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GMT OFF | 85.6188 | 86.1685 | 85.0745 | 94.9401 | 89.1013 | 67 | 14 |
| Learnable Threshold | 86.2227 | 86.2075 | 86.2393 | 95.5757 | 90.6788 | 40 | 12 |
| Generic MLP | 27.2625 | 87.4113 | 8.5130 | 29.4305 | 26.4818 | 1,389 | 15 |
| Full JEV | 16.8821 | 87.4940 | 3.2927 | 12.7108 | 10.0860 | 1,732 | 15 |

### Delta versus the same GMT OFF replay

| Method | ΔHOTA | ΔAssA | ΔIDF1 | ΔMOTA | ΔIDSW | ΔFrag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Learnable Threshold | +0.6038 | +1.1648 | +0.6355 | +1.5774 | -27 | -2 |
| Generic MLP | -58.3563 | -76.5615 | -65.5097 | -62.6195 | +1,322 | +1 |
| Full JEV | -68.7368 | -81.7818 | -82.2293 | -79.0153 | +1,665 | +1 |

The controller action-rate diagnostic explains the collapse:

| Method | ACCEPT | REASSOCIATE | START_NEW | WRITE_MEMORY | wrong-commit rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| GMT OFF | 48.849% | 0.000% | 1.151% | 50.000% | 9.241% |
| Threshold | 49.150% | 0.700% | 0.150% | 50.000% | 9.063% |
| Generic MLP | 15.008% | 0.000% | 34.992% | 50.000% | 35.696% |
| Full JEV | 0.000% | 14.782% | 35.218% | 50.000% | 48.810% |

The replay recorded one missing action-outcome lookup for Threshold and one
for Full JEV. These were retained as diagnostics and treated as zero
contamination contribution; they did not change the controller decision or
the GMT state. This is another reason the result is a screening warning and
not a paper claim.

## 4. Direct answers to the pilot questions

1. **Does the current H=8 data look reasonable?** Yes at the schema/state/
   utility level. It is not obviously degenerate, although ties are high.
2. **Can the three controllers learn?** They learn an offline predictive
   signal, but offline utility is effectively at the majority-policy ceiling;
   Full JEV is not superior. Runtime replay shows a serious semantic mismatch
   for MLP/JEV.
3. **Is there positive tracking signal?** Threshold shows a preliminary
   positive signal on `00021gate`; Generic MLP and Full JEV show catastrophic
   negative association signal. There is no positive Full JEV signal.
4. **Should we continue generating the remaining roughly one million records?**
   Do not wait blindly for a final controller result. The already-running
   formal builders remain untouched as required, but learned-policy expansion
   and final tracking selection should pause behind a runtime/offline parity
   investigation. First fix and unit-test feature/action/commit semantics on
   a tiny held-out subset, then rerun this pilot before using additional H=8
   data for final selection.

## 5. Immediate follow-up plan

1. Compare the exact feature vector and legal-action ordering emitted by
   `train_jev.py` with those emitted by `run_early_pilot_tracking.py` for the
   same record/context.
2. Add a checkpoint round-trip parity test: offline logits, decoded action,
   question type, and legal-action mask must match runtime for at least 50
   held-out contexts.
3. Investigate why runtime Full JEV selects no `ACCEPT_CURRENT` and why MLP
   selects `START_NEW` on about 35% of action opportunities despite their
   offline distributions.
4. Re-run only the small pilot after parity passes. Do not run the official
   full `VISION_test` or spend the remaining H=8 data on final selection until
   the learned controller no longer collapses association.

## Reproducibility artifacts

The repository copy contains the small JSON summaries and the source scripts.
The full JSONL snapshot, policy dataset, detector cache, prediction arrays,
and model checkpoints remain in the runtime directory and are intentionally
not committed to GitHub.

- `PILOT_DATA_AUDIT.json`
- `PILOT_OFFLINE_THREE_WAY.json`
- `PILOT_TRACKING_RAW.json`
- `PILOT_TRACKING_THREE_WAY.json`
- `PILOT_POLICY_SPLIT.json`
- `PILOT_SNAPSHOT_MANIFEST.json`
