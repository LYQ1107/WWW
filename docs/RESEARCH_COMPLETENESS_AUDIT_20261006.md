# Research completeness audit — 2026-10-06

Status: `IN_PROGRESS`

This is a state audit, not a final scientific result.  Final JEV claims remain
locked until the corrected-v4 provenance, parity, training, formal chunk, and
mutated-state tracking gates pass.

## Verified foundations

- Stage1 final checkpoint is present at
  `/data1/liuyeqiang/WWW/outputs/stage1_single_gpu/model_16000.pth`.
- Stage2 canonical checkpoint is present at
  `/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth`.
- The canonical Stage2 checkpoint SHA256 is
  `cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8`.
- The GMT environment reports Python 3.10.18, Torch 2.0.0+cu118, CUDA 11.8,
  and 10 visible GPUs.
- Existing Stage1/Stage2 validation, reload, optimizer-state, and finite-parameter
  evidence is recorded in the earlier research reports.  It is not being
  rerun or overwritten by the corrected-v4 data build.

## Corrected-v4 state

The v6/v7 canonical rebuild is pinned to source commit
`75b0aeaa7bfd3caaae942bc57b001c7c61edc8e4`; the video1 continuation includes
the later stale-bank promotion fix from the current branch.

- corrected v6 is still running under
  `/home/liuyeqiang/WWW_jev_rng_v4_runtime/small_h8_rng_controlled_v4_current_head_75b0aea`;
- corrected v7 is complete (3337 records) under the same root;
- corrected video1 continues under
  `/home/liuyeqiang/WWW_jev_rng_v4_runtime/small_h8_rng_controlled_v4_current/video_01`;
- corrected canonical-feature training is PASS under
  `/home/liuyeqiang/WWW_jev_rng_v4_runtime/small_h8_training_v4_canonical_features`;
  its report is `CORRECTED_V4_SMALL_H8_THREE_WAY_VIDEO06_VIDEO07.json` and uses
  seed `20261003`;
- same-code formal video7 single/chunk work is running under
  `/home/liuyeqiang/WWW_jev_rng_v4_runtime/formal_same_code_video7_v9_head`.

The earlier v6/v7 artifacts built from `5cd989d` remain diagnostic only.  The
old three-way report is preserved as
`reports/JEV_RNG_V4/SMALL_H8_THREE_WAY_VIDEO06_VIDEO07_LEGACY_SMALL_GATE_SCREENING_ONLY.json`.

## Gate inventory

| Gate | Current state | Evidence policy |
| --- | --- | --- |
| Stage1/Stage2 checkpoint | PASS | Existing validation/reload manifests and checkpoint SHA |
| Frozen GMT baseline | FROZEN | Reuse the approved canonical row; do not rerun full baseline |
| corrected v6/v7 records | v7 PASS / v6 RUNNING | Require source commit `75b0aea`, complete manifests, schema and SHA checks |
| corrected three-way training | PASS (canonical-feature v6/v7) | 8501 records, sequence-disjoint split, seed `20261003`, calibration and aggregate complete |
| runtime feature parity | WAITING | Three repeated runs, tolerance `2e-5`, exact structural fields |
| reactivation candidate parity | WAITING FOR VIDEO1 | Bounded native v3 trace is video1-only; exact IDs/order/action/event coverage remains required |
| formal single/chunk equivalence | RUNNING | Same source/checkpoint/cache/trace/RNG provenance and exact 3337 records |
| corrected video1 full parity | WAITING | MATCH, MEMORY, REACTIVATION, and TOTAL reported separately |
| final mutated-state tracking | BLOCKED BY GATES | No final checkpoint/runtime comparison before all gates pass |
| Full 24-video H=8 chunked rebuild | NOT AUTHORIZED | Formal same-code equivalence must PASS first |

Reports marked legacy, diagnostic, preliminary, or source-mixed are retained for
forensics but must not be used as final paper metrics.
