# Current-head dataset binding audit — 2026-10-07

这份只读审计解释为什么此前的正向 JEV screening 结果不能和最新 continuation 结果合并。两次使用的 source commit、records、compact dataset 和 calibrated checkpoints 不同，因此不是同一实验的重复运行。

## 关键事实

| 项目 | current-head | prior positive bundle |
|---|---|---|
| source commit | `1d2711e8` | `ec40effd` |
| policy records | 8496 | 8501 |
| dataset SHA | `3a0c5b05...f5df5` | `cabedc4f...7565c` |
| video06 records | 5162 | 5164 |
| video07 records | 3334 | 3337 |

按 `(video, frame, view, question, detection_index)` 对齐后，legal-action 集合在共同记录上完全一致，但 state feature 和 rollout return 并不一致：

- video06：5161 条共同记录中 4899 条 feature 不同，最大绝对差 `2843.3881`；270 行 target probability 不同。
- video07：3329 条共同记录中 3026 条 feature 不同，最大绝对差 `1685.5165`；200 行 target probability 不同。

所以旧 bundle 的 JEV 正向结果不能证明 current-head bundle 也应为正向；把两者合并会造成 source/data leakage 式的错误归因。

## current-head 数据的训练信号

current-head validation 上 majority action 的 accuracy `0.999806`，略高于 JEV `0.999613`；majority utility `36.629504` 也略高于 JEV `36.628971`。全体 best-action tie rate 为 `51.86%`，MEMORY 决策 tie rate 为 `100%`。这说明当前小数据集的监督信号高度退化，不能把“JEV beats Threshold/MLP”直接解释为有意义的 learned advantage。

current-head video01 closed-loop 已做同绑定第二次 replay，四方法指标、action counts、decision SHA 和 prediction SHA 全部一致；因此当前 `PILOT_NO_GO_CURRENT_HEAD_CONTROLLER_BUNDLE` 是稳定结论，而不是随机波动。

完整机器可读证据：[`CURRENT_HEAD_DATASET_BINDING_AUDIT_20261007.json`](../reports/JEV_RNG_V4/CURRENT_HEAD_DATASET_BINDING_AUDIT_20261007.json)。

Full H8 仍保持 `FULL_H8_AUTHORIZED=false`。
