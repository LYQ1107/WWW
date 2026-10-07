# JEV/GMT research live progress report

**Snapshot time:** `2026-10-07T01:51:56Z`  
**Repository:** `LYQ1107/WWW`  
**Branch:** `jev/counterfactual-rng-isolation-v4-20261006`  
**Snapshot commit before this report:** `378394b`  
**Classification:** live progress snapshot; **not a final research result**

This report records the state that was actually observed on the host. It
deliberately separates completed screening evidence from the still-pending
formal tracking claim.

## Executive summary

### Completed

1. GMT Stage1 and Stage2 checkpoints exist and were verified as resumable
   checkpoint files.
2. The canonical GMT OFF baseline is frozen and will be reused; the full OFF
   baseline is not being rerun.
3. The corrected-v4 RNG-isolation, reactivation-semantics, runtime-feature,
   stress-test, and current-head regression artifacts have been produced.
4. A current-head formal video07 single-worker versus three-chunk equivalence
   gate passed exactly for all `3337` semantic records. This authorizes the
   *method* of deterministic chunking for the future canonical H8 build.
5. Learnable Threshold, Generic MLP, and Full JEV each completed one offline
   screening training run on the corrected video06/video07 compact dataset.

### Still incomplete

- The corrected video01 artifact and its provenance/parity/stability gates are
  still running or pending.
- No corrected-v4 closed-loop GMT OFF / Threshold / MLP / JEV tracking result
  has been produced yet.
- Full H8 is still generating; no one of the 24 official shard manifests has
  been published at this snapshot.
- Full-H8 audited finalization, compact dataset, final sequence-disjoint
  training, Oracle comparison, official tracking evaluation, delta table, and
  final Go/No-Go report are not complete.

## Frozen inputs and checkpoints

| Item | Status | Evidence |
|---|---|---|
| GMT Stage1 | Complete and verified | `outputs/stage1_single_gpu/model_16000.pth` |
| GMT Stage1 SHA256 | `143e84deb50bdf5379c8f4463f1b9b237132e9281726b9cff469aff8c9dbe64e` | checkpoint audit |
| GMT Stage2 | Complete and verified | `outputs/stage2_single_gpu/model_20000.pth` |
| GMT Stage2 SHA256 | `cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8` | checkpoint audit |
| Training seed for first round | `20261003` | fixed seed policy |
| Formal H8 trajectory RNG master seed | `20261006` | partition and manifest provenance |
| H8 horizon | `8` | scheduler lock and record contract |

The checkpoint files are not being overwritten by the current H8 data build.

## Frozen GMT OFF baseline

These values are frozen from the canonical `model_20000.pth` and are reused as
the baseline row. A complete VISION_test OFF baseline is not being rerun.

| Metric | Frozen GMT OFF |
|---|---:|
| HOTA | 67.442 |
| DetA | 66.278 |
| AssA | 68.992 |
| IDF1 | 82.239 |
| MOTA | 80.942 |
| IDSW | 3092 |
| Frag | 8004 |
| CVIDF1 | 79.0248 |
| CVMA | 80.9276 |

These are frozen reference values, not a new measurement from the current
partial build.

## Correctness and implementation gates already passed

| Gate | Result | Scope |
|---|---|---|
| Corrected video06 provenance | PASS | 5162 records |
| Corrected video07 provenance | PASS | 3334 records |
| video06 runtime feature parity | PASS | 3 repetitions, no missing records, no OFF-action mismatch |
| video07 runtime feature parity | PASS | 3 repetitions, no missing records, no OFF-action mismatch |
| Numerical envelope | PASS for the locked gate | absolute `2e-5`; relative `4 * eps32` for explicitly scale-sensitive features |
| Formal video07 chunk equivalence | PASS | 3337/3337 semantic keys, zero missing/extra/mismatched records, identical canonical SHA and RNG provenance |
| JEV contract stress suite | PASS | legal masks, action permutation, threshold boundary, no future-GT runtime import, state sensitivity |
| CPU regression suite | PASS | `6 passed`, one existing Pillow deprecation warning |

The formal chunk gate is a correctness gate for the chunking implementation;
it is not evidence that the entire 24-video dataset is already correct.

Primary evidence files:

- `reports/JEV_RNG_V4/FORMAL_GMT_INTRA_VIDEO_CHUNK_EQUIVALENCE_VIDEO07_CURRENT_HEAD_V10.json`
- `reports/JEV_RNG_V4/RUNTIME_FEATURE_PARITY_VIDEO06_CURRENT_HEAD.json`
- `reports/JEV_RNG_V4/RUNTIME_FEATURE_PARITY_VIDEO07_CURRENT_HEAD_TOL2E5.json`
- `reports/JEV_RNG_V4/JEV_STRESS_CONTRACT_20261007.json`
- `reports/JEV_RNG_V4/CURRENT_HEAD_REGRESSION_TESTS_20261006.json`

## First three-model screening experiment

This experiment is complete only as an **offline screening run**. It used:

- corrected video06/video07 data;
- `8501` records;
- sequence-disjoint policy split;
- one seed, `20261003`;
- 20 epochs, batch size 128, AdamW, learning rate `0.001`;
- identical state, legal-action masks, targets, sample weights, optimizer and
  learning-rate settings for all three methods.

| Method | Val NLL | Best-action accuracy | Brier | ECE | Validation utility |
|---|---:|---:|---:|---:|---:|
| Learnable Threshold | 0.895639 | 0.916925 | 0.009966 | 0.479910 | 36.624177 |
| Generic MLP | 0.896707 | 0.704880 | 0.010381 | 0.284270 | 36.467661 |
| Full JEV | 0.894535 | 0.999806 | 0.008881 | 0.572758 | 36.644655 |

The majority-action reference has accuracy `0.999806` and validation utility
`36.644655`, exactly matching Full JEV's utility. Full JEV is therefore not
allowed to be described as a meaningful learned improvement at this stage.
Its apparent perfect accuracy is dominated by the same action distribution
as the majority reference. Multi-seed robustness was intentionally deferred
under the current first-round seed policy.

Evidence:
`reports/JEV_RNG_V4/CORRECTED_V4_SMALL_H8_THREE_WAY_VIDEO06_VIDEO07.json`.

## Live Full H8 data build

### Why it was started before video01 finished

The current Full H8 scheduler was authorized after the current-head formal
video07 single-worker/three-chunk equivalence gate passed. That gate had:

- `3337/3337` exact semantic records;
- zero `only_single`, `only_chunked`, or record mismatches;
- identical canonical SHA;
- exact checkpoint, trace, cache, source-commit, state-schema, utility, and
  trajectory-RNG provenance.

This was a wall-clock overlap decision: the long 24-video data build was
started while the independent corrected video01 gate continued. It does **not**
mean video01 passed, and it does **not** promote the partial H8 output to a
final dataset. The launch record explicitly classifies the run as
`AUTHORIZED_CANONICAL_FULL_H8_BUILD_NOT_FINAL_RESULT`.

The stricter risk boundary remains in force: no partial H8 records may be used
for final model selection or official tracking claims until all shard manifests,
provenance audits, compact conversion, and the required runtime gates pass.

### Queue snapshot

At snapshot time, the scheduler reported `RUNNING` with `1,112,173` source
decision units. The visible line counts below are temporary JSONL prefixes,
not official manifest record counts; workers are still appending to them.

| Video | Status | Visible lines / source units | Worker |
|---:|---|---:|---|
| 3 | RUNNING | 657 / 70,422 | gpu4-slot2 |
| 11 | RUNNING | 413 / 30,239 | gpu6-slot3 |
| 12 | RUNNING | 445 / 29,104 | gpu9-slot3 |
| 13 | RUNNING | 449 / 33,845 | gpu4-slot3 |
| 14 | RUNNING | 695 / 64,571 | gpu6-slot2 |
| 15 | RUNNING | 4,039 / 150,103 | gpu8 |
| 16 | RUNNING | 2,379 / 116,801 | gpu6 |
| 19 | RUNNING | 479 / 31,580 | gpu8-slot3 |
| 20 | RUNNING | 671 / 48,083 | gpu8-slot2 |
| 22 | RUNNING | 624 / 41,633 | gpu9-slot2 |
| 23 | RUNNING | 4,022 / 180,321 | gpu4 |
| 24 | RUNNING | 3,892 / 139,851 | gpu9 |
| **Total visible prefix** |  | **18,765 / 1,112,173** |  |

Pending videos at this snapshot are `1, 2, 4, 5, 6, 7, 8, 9, 10, 17, 18,
21`. No video has a formal `COMPLETE` manifest yet.

Resource policy:

- active Full H8 workers use only the authorized safe GPUs 4, 6, 8 and 9;
- reserved/foreign GPUs 2, 3, 5 and 7 are not used;
- the corrected video01 builder is separate and has not been stopped or
  migrated;
- no frozen baseline or full VISION_test OFF inference is being rerun.

### Partial-prefix validation already performed

A read-only strict audit of the currently visible temporary records completed
successfully for `18,377` complete JSONL lines (the queue grew while the audit
was running). Every audited line passed:

- JSON parsing and `validate_record(..., allow_future_gt=True)`;
- H8 horizon check;
- canonical Stage2 checkpoint SHA check;
- state digest check;
- legal-action and action-outcome consistency;
- target-probability normalization;
- finite state features;
- video-context consistency.

This is useful evidence against an immediate structural failure, but it is not
a proof of whole-dataset semantic correctness. The final authority remains
the complete per-shard manifest/provenance/trace audit and the independent
video01 runtime gates.

## Currently running and waiting work

- Corrected video01 builder: PID `8251`; output is not yet complete.
- Original Full H8 scheduler: PID `19950`.
- Additional supervised Full H8 slot workers: PIDs `40330`, `40345`,
  `40354`, `40364`, `7418`, `7425`, `7439`, `7453`.
- Corrected video01 tracking waiter: PID `19317`, fail-closed.
- Full H8 aftercare/official-evaluation waiters are alive and consume no GPU
  while prerequisites are missing.

The waiters are deliberately fail-closed: they do not turn a missing or
failed manifest into a usable result.

## Pending gates and final order

1. Finish and atomically validate corrected video01 (`8995` expected records).
2. Run video01 provenance, MATCH/MEMORY/REACTIVATION feature parity,
   native-vs-replay candidate parity, and repeated numerical stability.
3. Run the corrected-v4 closed-loop GMT OFF / Threshold / Generic MLP / Full
   JEV comparison on the held-out video01 sequence, if those gates pass.
4. Let all 24 Full H8 shards finish; require every manifest to pass schema,
   provenance, record-count, hash, state, RNG and duplicate checks.
5. Run audited full-H8 finalization, compact conversion, and the fixed
   sequence-disjoint TRAIN/VAL split.
6. Train the three methods once with seed `20261003`, then run the authorized
   Oracle and formal tracking evaluation using the same frozen GMT checkpoint,
   detector/perception cache, mapping fix and evaluation protocol.
7. Publish absolute metrics, deltas against frozen GMT OFF, failure/gate
   status, environment details, and reproducibility hashes.

## Current Go/No-Go conclusion

The authoritative status remains:

```text
status   = IN_PROGRESS
decision = NO_FINAL_RESEARCH_CLAIM_YET
```

The project has completed substantial implementation and screening evidence,
but it has **not** completed the formal corrected-v4 tracking comparison or
the full-H8 official evaluation. No positive JEV tracking claim is justified
yet.

Authoritative status file:
`reports/JEV_RNG_V4/CURRENT_GO_NO_GO_20261007.json`.
