# JEV Recovery 有界实验最终科学复核（2026-10-14）

本文件是对 2026-10-12 recovery 分支已完成有界执行的最终复核。它不替换原始预注册、checkpoint、Phase V–XVI 结果或较早的中间报告；较早报告保留为时间点快照。

## 判定

**`DEVELOPMENT_NOT_ESTABLISHED` / `NO_GO_FOR_JEV_INDEPENDENT_ADVANTAGE`**。

本轮已经完成 R1 pilot、R2 XIV formal、R3 XIII formal 的冻结路线与同条件对照，但没有证明 JEV 比 Pair Threshold、普通 MLP 或 Ordinary Attention 具有稳定的独立在线身份收益。`MASTER_BREAKTHROUGH=false`，原有 master goal gate 未被清除。

## 最关键的身份关联结果

所有正式 online 数字来自同一 `FULL_ONLINE_RESULTS.json` 的 native 三种子汇总；不是从单个 DEV 视频挑选。

| 路线 | 方法 | HOTA | AssA | CVIDF1 | IDSW |
|---|---|---:|---:|---:|---:|
| R2 XIV | Strong parent | 73.505 | 73.639 | 88.546 | 391.7 |
| R2 XIV | Pair Threshold | 72.355 | 71.371 | 86.047 | 663.7 |
| R2 XIV | MLP | 71.269 | 69.269 | 84.471 | 898.0 |
| R2 XIV | Ordinary Attention | 73.576 | 73.796 | 88.515 | 774.7 |
| R2 XIV | JEV | 73.026 | 72.810 | 88.057 | 1462.3 |
| R3 XIII | Strong Fixed parent | **75.637** | **77.907** | **91.889** | **105.0** |
| R3 XIII | Pair Threshold | 73.738 | 74.202 | 87.808 | 257.3 |
| R3 XIII | MLP | 69.314 | 65.496 | 79.795 | 813.7 |
| R3 XIII | Ordinary Attention | 71.367 | 69.441 | 83.691 | 978.7 |
| R3 XIII | JEV | 69.120 | 65.215 | 80.543 | 1038.0 |

R2 中 JEV 的 HOTA 高于 Threshold 0.671，但 IDSW 多 798.6；低于 Ordinary Attention 0.550 HOTA。R3 强 Fixed 基座下，JEV 比 Threshold 低 4.618 HOTA、低 8.987 AssA、低 7.265 CVIDF1，同时多 780.7 IDSW；比原 Strong Fixed 低 6.517 HOTA并多 933.0 IDSW。因此不能把 R2 的局部 HOTA 差异解释成 JEV 的独立身份优势。

## 两个辅助问题为什么没有自动转化成 Global ID 收益

冻结同一个 JEV checkpoint、真实 TRAIN16 状态和合法候选集合的 2067 个状态诊断显示：

- 完整 JEV 的认证 N01/N10 为 238/131，净纠错 +107；只保留 Cross 为 51/52，净 -1；只保留 Persistent 为 41/9，净 +32；简单平均为 11/14，净 -3；去掉 joint-plan residual 仍为 238/131，说明这批状态的净值不能归因于该 residual。
- 同一批次的普通 Threshold、MLP、Attention 净纠错分别为 +119、+151、+147；JEV +107 没有独立领先。
- 两个辅助问题都预测正确但最终身份错误的事件有 663 个；其中仅 188 个存在已认证且已进入方案的正确 active-ID 替代，475 个没有可认证的正确方案替代。另有 1883 行已认证正确 active-ID 没有进入最终方案，302 行没有任何可认证正确替代。因此辅助分类准确率不能直接当作身份纠错失败或成功。
- 将辅助 logits 置零时最终评分逐位不变，说明当前最终评分没有直接读取两个 Head 的输出；联合表示仍可能间接影响决策，但“Head 准确率高”本身不是因果证据。
- H64 配对 native 窗口中，完整 JEV 的已知错误观测为 286；Cross-only 182、Persistent-only 181、Mean 182。冻结权重干预没有形成稳定的全局在线优势。

这些结果同时支持 A/B/C 三类解释：辅助目标与最终候选区分并不等价；输出融合存在直接路径缺失/分布改变问题；候选覆盖、认证状态和 UNKNOWN 使一部分样本没有合法可评价的正确动作。不能把原因简化为训练步数不足，也没有事后重写 R2 标签或 Gate。

## R2/R3 和资源边界

- 所有正式 native online arm 均为三种子；identity 审计共 95 个正式 case，效率包 15 个方法源。
- Stage2 的正式组件 profile、参数读取和 full-image restart FPS 已保留；profile 中的 nested timing 与 Gallery/Bank Python residual 分开记录，没有把部分 FLOPs 当成完整系统 FLOPs。
- R2/R3 队列自然结束，监控 `live_jobs=[]`、`unreviewed_failures=[]`；不启动 Full24，不读取 official TEST，不触碰封存的 VisionTrack20/21/22。
- `labels_modified=false`、`weights_modified=false`、`official_TEST=false`、`sealed20_21_22=true` 均在诊断绑定中保留。

## 下一步边界

本轮不再用 DEV17/18/19 反复调参，也不把共同的候选覆盖收益包装成 JEV 结构收益。若要继续，必须以新的预注册实验 ID 明确处理候选条件化的真实合法动作覆盖、直接可审计的身份提交和独立完整视频；在此之前，当前 MATCH 架构不应被宣称已经优于传统关联控制器。

复核输入：

- `reports/JEV_RECOVERY/FULL_ONLINE_RESULTS.json`
- `reports/JEV_RECOVERY/QUESTION_TO_ASSOCIATION_DIAGNOSIS.json`
- `reports/JEV_RECOVERY/PARAMETERS_AND_FPS.json`
- `reports/JEV_RECOVERY/finish/READY.json`（runtime，仅作完成状态绑定）
- `reports/JEV_RECOVERY/R3_JEV_INDIVIDUAL_MECHANISM_REVIEW.md`
