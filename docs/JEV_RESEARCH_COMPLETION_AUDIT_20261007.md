# WWW research completion audit — 2026-10-07 (updated)

This is a live completion audit, not a final paper-results claim. It records
the latest reproducible state after the corrected video01 v2 run and the
current-head segmented small-gate closed loop.

## Current decision

`INCOMPLETE_NO_GO_CURRENT_HEAD_CONTROLLER`

The formal video01 artifact completed and passed file/provenance checks, but
its runtime-semantic aftercare failed. The current-head controller bundle also
has a stable small-gate NO-GO: Full JEV is repeatable but its held-out
association metrics are below the same-run GMT OFF baseline, while Threshold
and MLP degrade more strongly. Therefore no final paper claim and no Full H8
authorization are allowed.

## Verified complete

- GMT Stage1 checkpoint validation: PASS.
- GMT Stage2 `model_20000.pth` validation: PASS. The independent validation
  checked iteration 20,000, scheduler epoch 20,000, 411 model keys, finite
  tensors, optimizer presence, and checkpoint reload.
- Stage1 SHA-256:
  `143e84deb50bdf5379c8f4463f1b9b237132e9281726b9cff469aff8c9dbe64e`.
- Stage2 SHA-256:
  `cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8`.
- The frozen GMT baseline was reused; no complete VISION_test baseline rerun
  was performed for this gate.
- Corrected video06/video07 provenance, runtime parity, repeatability, RNG
  branch isolation, proposal-reuse, and bounded chunk-equivalence checks
  passed.
- The corrected segmented small gate completed its acceptance boundary:
  `17491/17491` records, `88/88` chunks, zero pending/running chunks. Its
  bounded early/late equivalence checks were exact.
- Current-head Threshold/MLP/JEV training and closed-loop execution completed
  with seed `20261003`; exact repeatability of metrics, action counts, and
  decision/prediction hashes passed.

## Formal video01 v2 result

The corrected formal video01 builder ran in one process:

```text
PID:       12163
started:   2026-10-07 04:15:18 UTC
completed: 2026-10-07 11:10:35 UTC
elapsed:   06:55:17.54
records:   8995/8995
records SHA-256: 1da2ed9e7eec43c7cf139754d873f78a299c344ff74d65c6866b92f18f3da79
source commit: 4108f18f5040432f68d55872e81e9f76e9acd08f
```

The artifact is complete and the manifest/provenance checks pass. The
scientific hard gates do not:

- runtime feature parity: FAIL; max absolute error `0.006004035472869873`
  versus tolerance `2e-5`;
- reactivation candidate parity: FAIL; 225 mismatches, including candidate
  score and selected-proposal differences;
- runtime feature stability: FAIL across three repetitions;
- corrected video01 closed loop: NOT RUN because the pre-gates failed.

The machine-readable six-hour evidence is
[`VIDEO01_SIX_HOUR_RUNTIME_EVIDENCE_20261007.json`](../reports/JEV_RNG_V4/VIDEO01_SIX_HOUR_RUNTIME_EVIDENCE_20261007.json).

## Current-head small-gate result

The latest controller bundle is bound to source commit
`1d2711e80ac5fa00806fd9eed30e90cd51df6b30`, policy dataset SHA
`3a0c5b05956d62ea09837678261559ac94d65ba77ba8eb16710c183f724f5df5`, and
the 8,496-record video06/video07 policy dataset.

On held-out video01, same-run GMT OFF versus current-head JEV was:

| Method | HOTA | AssA | IDF1 | MOTA | IDSW |
|---|---:|---:|---:|---:|---:|
| GMT OFF | 86.523 | 85.149 | 94.642 | 89.579 | 282 |
| Full JEV | 85.072 | 82.460 | 91.975 | 95.586 | 18 |

Thus JEV deltas were `ΔHOTA=-1.451`, `ΔAssA=-2.689`,
`ΔIDF1=-2.668`, `ΔMOTA=+6.007`, and `ΔIDSW=-264`. Threshold and MLP were
also below GMT OFF. This is a stable screening NO-GO, not a final paper
comparison.

An earlier positive screening result is retained as a separate legacy
controller/data binding. It is not a repeatability replicate and must not be
merged with the current-head result.

## Active jobs and Full H8 status

- No corrected video01 builder, segmented small-gate builder, or closed-loop
  job is active at this audit snapshot.
- The earlier speculative Full H8 run is paused, noncanonical, and retained
  only for diagnostics/speed profiling. Its records cannot be merged, used for
  policy training, or reported as paper metrics.
- Full H8 authorization remains `false` because the corrected video01 gates
  and the current-head controller gate are not both GO.

## Required next work

1. Diagnose the current-head data/training degeneration and the learned-policy
   runtime state shift; do not merge the legacy positive bundle.
2. Apply the smallest justified interface/data correction and rerun a fresh
   held-out small gate with a new source/data/checkpoint binding.
3. Require runtime parity, candidate parity, stability, and closed-loop GO
   before any full-data authorization.
4. If and only if that GO is obtained, create a frozen H8 worktree, enforce
   worker/scheduler commit hard gates, rebuild the 24-video canonical dataset,
   and then run final training/evaluation.

## Review links

- [audit-diagnostics-20261007](https://github.com/LYQ1107/WWW/tree/jev/audit-diagnostics-20261007)
- [current-head closed-loop result](JEV_CURRENT_HEAD_CLOSED_LOOP_RESULT_20261007.md)
- [current-head dataset binding audit](JEV_CURRENT_HEAD_DATASET_AUDIT_20261007.md)
- [video01 six-hour evidence](JEV_V2_LIVE_PROGRESS_AND_SIX_HOUR_EVIDENCE_20261007.md)
