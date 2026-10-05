# GMT VisionTrack inference-config audit

状态：进行中（baseline anomaly unresolved）。比较基线为 repository 初始
official source commit `8fa96705e86f74dcc880b1cf332e8e448ac3ef0f`，当前审计
worktree 为 `dfbbdb8`。所有差异先记录，不静默修改 official evaluation。

## YAML recipe comparison

`configs/VISION_test.yaml`、`configs/VISION_stage1.yaml` 和
`configs/VISION_stage2.yaml` 与 initial repository commit 的文本/字段比较
目前为 `IDENTICAL`。因此下列字段没有发现 current-vs-released YAML 差异：

| Field group | Result |
| --- | --- |
| `ASSO_THRESH`, `ASSO_THRESH_TEST`, `ASSO_WEIGHT` | IDENTICAL |
| `WITH_BANK`, `THRED`, `BANK_SIZE` | IDENTICAL |
| `OVERLAP_THRESH`, `NOT_MULT_THRESH`, `MIN_TRACK_LEN` | IDENTICAL |
| `VIDEO.TEST_LEN`, `INPUT.TEST_SIZE`, `INPUT.TEST_INPUT_TYPE` | IDENTICAL |
| `WITH_IOU`, `DECAY_TIME`, `MAX_CENTER_DIST` | IDENTICAL |
| solver optimizer, LR, batch, MAX_ITER | IDENTICAL in the three compared YAML files |
| dataset train/test names and source-aware loader | IDENTICAL |

The current VISION test YAML still contains `TEST_SIZE=1560`,
`VIDEO_TEST.MIN_TRACK_LEN=50`, `ASSO_HEAD.THRED=0.4`,
`ASSO_HEAD.WITH_BANK=True`, and the released VisionTrack test association
settings. No threshold was changed as part of this audit.

## Source-level additions that are not released YAML fields

Current `gtr/config.py` adds a `MODEL.JEV` configuration namespace with
defaults such as `ENABLED=False`, `MODE='off'`, trace path and controller
weights. These defaults are `SOURCE_ADDITION`, not an inference YAML mismatch;
the initial repository did not define this namespace. The strict traced OFF
run explicitly overrides `MODEL.JEV.ENABLED=True`, `MODEL.JEV.MODE=off`, and a
trace path to collect evidence. This must be compared against the untouched
pre-JEV GMT source before it can be called semantically equivalent.

## Current strict-run overrides

The live strict traced command uses the same Stage2 checkpoint and VISION test
YAML, with only instrumentation-related overrides:

```text
MODEL.JEV.ENABLED=True
MODEL.JEV.MODE=off
MODEL.JEV.TRACE_PATH=<strict traced OFF trace>
```

The native strict run has no JEV runtime trace. The native/traced cache writer
paths are isolated after the cache collision fix in commit `f3187b8`.

## Pending source/config checks

- Compare pre-JEV `gtr/modeling/meta_arch/gtr_rcnn.py` at initial commit with
  current JEV-disabled code on a fixed TRAIN mini subset.
- Confirm `test_net.py`, mapper, postprocessing, and TrackEval preparation do
  not introduce a non-YAML inference mismatch.
- Extract checkpoint load warnings from the model-20000 inference log.
- Record the exact historical evaluation commit in the inference/evaluation
  manifest; the existing manifest has source overlay but no git commit field.

Until these checks finish, this document reports config parity only and does
not clear the baseline anomaly.
