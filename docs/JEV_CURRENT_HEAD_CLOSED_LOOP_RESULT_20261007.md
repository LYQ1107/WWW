# Current-head controller closed-loop result — video01 (2026-10-07)

这是 continuation supervisor 使用最新 segmented video06/video07 训练 bundle 后，对 corrected video01 做的 screening-only 闭环。它与此前正向 JEV 报告使用的 controller bundle 不同，不能合并或互称重复实验。

## 绑定

- source commit：`1d2711e80ac5fa00806fd9eed30e90cd51df6b30`
- video01 records：8995，SHA `a174399cb979d5d1f9dc41366a4f07f6299c159c68e10e19222fedc656db7611`
- GMT checkpoint SHA：`cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8`
- policy dataset records：8496，manifest SHA `3a0c5b05956d62ea09837678261559ac94d65ba77ba8eb16710c183f724f5df5`
- policy split SHA：`bbdac3cff7c46b20c707dd278c526704a99ede973cc1ffde31b11172df9efd91`
- seed：`20261003`

## Tracking 结果

| 方法 | HOTA | AssA | IDF1 | MOTA | IDSW | ΔAssA | ΔIDSW |
|---|---:|---:|---:|---:|---:|---:|---:|
| GMT OFF | 86.523 | 85.149 | 94.642 | 89.579 | 282 | 0.000 | 0 |
| Learnable Threshold | 80.034 | 72.917 | 83.995 | 78.453 | 771 | -12.232 | +489 |
| Generic MLP | 68.217 | 53.514 | 72.876 | 72.833 | 1018 | -31.635 | +736 |
| Full JEV | 85.072 | 82.460 | 91.975 | 95.586 | 18 | -2.689 | -264 |

所有 execution/parity/candidate/stability gate 均完成；但当前 runner 的 tracking verdict 是 `PILOT_FAIL_RUNTIME_STATE_PARITY_OR_TRACKING`，因为 JEV 的 AssA 低于同一 video01 GMT OFF baseline。因而：

```text
PILOT_NO_GO_CURRENT_HEAD_CONTROLLER_BUNDLE
FULL_H8_AUTHORIZED = FALSE
```

## 与上一份正向 JEV 结果的关系

上一份 JEV 正向结果使用 dataset SHA `cabedc4f...7565c` 及另一组 calibrated checkpoints；本次使用 dataset SHA `3a0c5b05...f5df5` 和新的 calibrated checkpoints。两者不是同一 bundle 的 repeatability，不能把指标差异归因于随机波动。本次 current-head bundle 已做第二次完整 replay：四个方法的指标、action counts、决策 JSON SHA 和 prediction JSON SHA 均逐项一致，repeatability 为 `PASS`。因此当前 NO-GO 是稳定结果；Full H8 继续保持暂停。

机器可读报告：[`SEGMENTED_SMALL_GATE_CURRENT_HEAD_CLOSED_LOOP_VIDEO01_20261007.json`](../reports/JEV_RNG_V4/SEGMENTED_SMALL_GATE_CURRENT_HEAD_CLOSED_LOOP_VIDEO01_20261007.json)。数据绑定差异审计：[`CURRENT_HEAD_DATASET_BINDING_AUDIT_20261007.json`](../reports/JEV_RNG_V4/CURRENT_HEAD_DATASET_BINDING_AUDIT_20261007.json)。
