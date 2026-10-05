# Current formal V2 state

Updated: 2026-10-05 UTC. This snapshot records evidence and process state; it
does not turn a running or diagnostic artifact into a completed experiment.

## Canonical checkpoint

- `/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth`
- SHA256: `cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8`
- validation JSON: `status=PASS`, iteration/scheduler/global iteration `20000`
- optimizer state `PRESENT`, model finiteness `PASS`, checkpoint reload `PASS`

Proxy `model_4500.pth` is screening evidence only. A final lock is required to
carry `CANONICAL_MODEL20000_ONLY` and the canonical checkpoint digest.

## Live processes at snapshot

| PID | Role | State |
| ---: | --- | --- |
| 10016 | original v2 pipeline parent | running in formal counterfactual phase |
| 10350 | original train formal shard 0, videos `1 5 9 13 17 21` | running |
| 29280 | manually started low-memory train shard 1, videos `2 6 10 14 18 22` | running |
| 33788 | pre-lock TEST shard 0 diagnostic, videos `1 5 9 13 17 21` | running; never used for selection |
| 16606 | fresh strict same-GPU OFF gate | running sequential native then traced on visible GPU 0 |

The existing formal processes are retained. New TEST counterfactual processes
are fail-closed until the canonical final lock and selection-protocol digest
exist; the already-running pre-lock diagnostic is retained and will be marked
`PRELOCK_TEST_DIAGNOSTIC_DO_NOT_USE_FOR_SELECTION` if it completes.

## Completed evidence

- Stage1/Stage2 checkpoint validation: PASS.
- Canonical OFF train/test/native inference manifests: COMPLETE and tied to the
  canonical checkpoint.
- Train/test perception caches and trace alignment: PASS.
- Historical cross-GPU OFF report: retained as
  `CROSS_GPU_STRUCTURAL_DIAGNOSTIC_ONLY`; it is not a formal gate.
- Canonical OFF TEST baseline metrics were calculated for reporting only; see
  `docs/STAGE2_OFF_TRACKING_RESULTS.md`.
- Selection and TEST-access invariant tests: PASS.

## Protocol gates still open

1. Fresh strict same-GPU OFF exact equality and v2 replay-equivalence report.
2. Complete formal TRAIN counterfactual shards and merged manifest.
3. Generate TRAIN-only H=1/8/16/32 datasets without reading TEST.
4. Run all threshold/nonlinear-threshold and generic-MLP controls with equal
   supervision, three seeds, and validation-only calibration.
5. Run feature audit, same-score/different-state analysis, reviewer controls,
   and automatic policy-val model/horizon selection.
6. Create the canonical final selection lock; only then generate/read official
   TEST counterfactuals and run official Oracle/baseline/JEV evaluation.

No GO/NO-GO conclusion is authorized yet.
