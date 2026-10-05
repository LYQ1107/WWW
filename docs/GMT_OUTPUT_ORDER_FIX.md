# VisionTrack output-order fix

状态：`IMPLEMENTED_AND_REGRESSION_PASS`

## 问题

VisionTrack 的多视角输入由 `GMTDatasetMapper` 按 view-block 排列：

```text
V1F1 V1F2 V1F3 V2F1 V2F2 V2F3
```

`GTRRCNN.sliding_inference_GMT` 返回 frame-major 输出：

```text
V1F1 V2F1 V1F2 V2F2 V1F3 V2F3
```

旧版 `MOTEvaluator.process()` 直接 `zip(inputs, outputs)`，因此 output 会被
写入错误的 `image_id`。这会污染后续 frame/view/sequence 评测标签，但不改变
GMT 的 detector 或 tracker 计算本身。

## 正式修复

`gtr/evaluation/mot_evaluation.py` 新增
`align_visiontrack_inputs_to_outputs()`，并在 `MOTEvaluator.process()` 中仅对
`VISION_train` / `VISION_test` 应用 view-block → frame-major 映射：

```text
output[k] → inputs[view * frames_per_view + frame]
```

单视角、非 VisionTrack 数据集、输入输出长度异常和不整除的多视角 batch
分别保持 identity 或显式报错。修复提交：`830c5cc`。

## Regression test

运行：

```bash
PYTHONPATH=third_party/CenterNet2 \
/home/liuyeqiang/anaconda3/envs/GMT/bin/python -u \
reproduction_tools/test_visiontrack_output_order_fix.py \
  --real-subset --video-ids 1 2 \
  --report /data1/liuyeqiang/WWW/outputs/research_final_v2/baseline_debug/output_order_fix_regression.json
```

结果：`PASS`。

- synthetic contract tests：3/3 PASS；
- real subset：video 1、2，共 5,330 images / 26,471 prediction rows；
- discrete fields：exact；
- serialized float fields：exact（强于 `1e-6` tolerance）；
- source-fixed evaluator labeling semantics 与此前 deterministic diagnostic
  remap：exactly equal；
- video 1 mapping changed：2,008 labels；video 2 mapping changed：3,318 labels。

回归报告：

`/data1/liuyeqiang/WWW/outputs/research_final_v2/baseline_debug/output_order_fix_regression.json`

报告中的固定输入 hashes：

- TEST annotations：`7a1e735bf25026a56148deaeb432e0520c6ad5dad97d5362aa58739255389593`；
- 原始 canonical prediction：`a8d2f5c4cc758a887276e59671831215d36a8949d07486a34d78076aaf3e7b3d`；
- prediction stream：`065777f02ff84d91da209f5dbefebb8a7ee78002a9b25b4cf86df88d2da3a8b1`。

## Baseline policy

本修复只完成 source-level label alignment regression，不重新执行完整
`VISION_test` baseline，也不重新训练 GMT。冻结的 canonical baseline 仍为：

```text
HOTA   67.442
DetA   66.278
AssA   68.992
IDF1   82.239
MOTA   80.942
IDSW   3092
Frag   8004
CVIDF1 79.0248
CVMA   80.9276
```

后续 Learnable Threshold、Generic MLP 和 Full JEV 必须共用同一份 H=8
counterfactual policy dataset、同一 split/监督/seed/训练预算；不得复制或重跑
完整 baseline trace/cache/predictions。
