# Segmented small-gate closed-loop result — video01 (2026-10-07)

## 结论

当前 corrected video01 的 provenance、三视频 runtime parity、reactivation candidate parity 和三次 stability 全部通过。用已有的 canonical-feature、single-seed controller 做完整 video01 closed-loop 后，Full JEV 出现明确正向 association 信号；Learnable Threshold 和 Generic MLP 则严重退化。

因此本轮结论是：

```text
PILOT_GO_FOR_JEV_CONTINUATION
PILOT_WARNING_FOR_THRESHOLD_AND_MLP
FULL_H8_AUTHORIZED = FALSE（本报告不单独授权）
```

这仍是 screening/diagnostic 结果，不是最终论文结果，也不是完整 VISION_test 结果。

## Tracking 指标

| 方法 | HOTA | AssA | IDF1 | MOTA | IDSW | Frag | ΔHOTA | ΔAssA | ΔIDF1 | ΔIDSW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GMT OFF | 86.523 | 85.149 | 94.642 | 89.579 | 282 | 6 | 0 | 0 | 0 | 0 |
| Learnable Threshold | 43.176 | 21.281 | 37.211 | 26.530 | 3053 | 6 | -43.347 | -63.868 | -57.431 | +2771 |
| Generic MLP | 19.044 | 4.179 | 12.598 | 4.869 | 4005 | 6 | -67.479 | -80.970 | -82.044 | +3723 |
| Full JEV | **88.925** | **90.022** | **97.893** | **95.768** | **10** | 6 | **+2.402** | **+4.873** | **+3.250** | **-272** |

JEV 的 wrong-commit rate 为 `0.00068`，memory contamination 为 `0`；Threshold/MLP 的 wrong-commit rate 分别为 `0.42695` 和 `0.66475`，MLP memory contamination 为 `78`。

## Gate 证据

- video01/06/07 runtime parity：全部 PASS，分别比较 8995/5162/3334 条，tolerance exceed 均为 0，OFF action mismatch 均为 0。
- video01 candidate parity：174 native events 对 174 replay events，mismatch `0`。
- video01 stability：3 次、每次 8995 条，numeric tolerance 全部 PASS。
- segmented chunk equivalence：early 70/70、late 102/102 exact；这是 bounded evidence，不替代正式 Full H8 authorization 的完整等价性要求。

机器可读完整汇总：[`SEGMENTED_SMALL_GATE_CLOSED_LOOP_VIDEO01_20261007.json`](../reports/JEV_RNG_V4/SEGMENTED_SMALL_GATE_CLOSED_LOOP_VIDEO01_20261007.json)。原始 tracking report 保存在本机，SHA-256 为 `943754233d997d1fc081ad4c73fcd21ce7c1bef75f14789e1f5812db66ee79cc`。

## 后续决策

JEV 满足当前 pilot runner 的 GO 条件（AssA 不低于 GMT OFF 且 IDSW 不高于 GMT OFF），但 Threshold/MLP 的崩溃不能被忽略。正式 Full H8 之前需要保留这项 warning，并明确是否将 video01 纳入新的 train/val 设计；同时仍要从唯一冻结 source commit 创建 canonical worktree，并完成完整 authorization gate。
