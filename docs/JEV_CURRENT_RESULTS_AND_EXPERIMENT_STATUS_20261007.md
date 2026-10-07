# JEV 当前运行结果与实验状态（2026-10-07）

本页是当前可审查的状态快照。GitHub 中保存的是报告、时间戳、源代码 commit 和 SHA-256 证据；原始 JSONL、模型和缓存仍保留在本机运行目录，没有把大文件伪装成论文结果。

## 结论先说

- corrected formal video01 H=8 v2 已真实运行约 **6 小时 55 分 17 秒**，完成并落盘 **8995/8995** 条记录。
- 文件和 provenance 完整性通过，但 runtime semantics gate 没通过：runtime feature parity、reactivation candidate parity、stability 均 FAIL。因此这批记录**不能作为最终训练数据或论文结果**。
- 当前 segmented small gate 已在 acceptance boundary 完成 `17491/17491` 条，随后完成了 current-head controller 的四路 video01 closed-loop；Full H8 保持暂停、非 canonical、未授权。
- 原 MPS driver 已 graceful 停止；当前没有 small-gate builder/closed-loop 进程。
- MPS 前后 bounded equivalence 已通过：早期 `70/70`、晚期 `102/102` 逐条 exact，canonical SHA 和字段比较均一致。实时 15.07 秒抽样增加 250 条，但不据此宣称稳定 ETA。

## 本次只读 evaluator 审计结果（不改变暂停边界）

在不重建 video01、不重训三模型、不启动 Full H8 的前提下，使用已有
`canonical_test_predictions_order_fixed.json` 做了 evaluator-only 复核。结果报告为
[`FIXED_EVALUATOR_OFF_CANDIDATE_20261007.json`](../reports/JEV_RNG_V4/FIXED_EVALUATOR_OFF_CANDIDATE_20261007.json)。

- 单视角 TrackEval：`PASS`；HOTA `67.442`、DetA `66.278`、AssA `68.992`、IDF1
  `82.239`、MOTA `80.942`、IDSW `3092`、Frag `8004`。
- 跨视角顺序时间轴：CVIDF1 `79.0248`、CVMA `80.9276`；交错时间轴 CVMA
  `74.2925`。
- 该结果与既有 output-order 修正诊断值一致；它不是新的 model inference，也不是
  新的 counterfactual/training 结果。
- VisionTrack 原始 GT 的重复 track-ID 行被保留，因此本次明确使用
  `--allow-duplicate-gt` audit mode，不能直接写成最终论文 strict 指标。
- video01 六小时证据仍然有效：`04:15:18Z` 至 `11:10:35Z`，耗时 `06:55:17.54`，
  `8995/8995` 条已落盘；但其 runtime feature parity、reactivation candidate parity
  和 stability 失败，科学结论仍是 `NO_GO_RUNTIME_SEMANTICS_GATE`。

因此当前真正的剩余工作仍是 branch-consistent state/feature audit、reactivation
candidate-level supervision 和新的 video01 runtime/closed-loop gate；本次 evaluator
结果没有授权重建 video01、重训或启动 Full H8。

## 最新补充指令再审计（14:43 UTC）

已重新读取并绑定当前补充指令文件，SHA-256 为
`84801e5312b0f8884937da16379487d03a5286421f8e58b648181120f69d6529`。
本次只做只读/有界诊断，没有启动第二个完整 video01 builder，也没有修改
已经完成的 v2 产物。完整再审计见
[`VIDEO01_DIAGNOSTIC_REAUDIT_20261007.json`](../reports/JEV_RNG_V4/VIDEO01_DIAGNOSTIC_REAUDIT_20261007.json)
和 [`JEV_V2_DIAGNOSTIC_REAUDIT_20261007.md`](JEV_V2_DIAGNOSTIC_REAUDIT_20261007.md)。

- **重算范围：** 4108f18 相对 60675dae 只改动 builder 的 11/4 行，但同输入
  20 条样本的 state 是 `20/20` 相同、outcome/label 是 `0/20` 相同；旧
  60675 回放也复现不了旧 8995 条 artifact。因此旧 outcomes、targets、
  best-actions、sample weights 全部禁止复用，必须完整重算。
- **candidate parity：** fresh v2-bound replay（至 frame 220）为 FAIL；5/5
  事件互相找到，但有 `2` 个 native-only、`2` 个 replay-only key；重叠事件
  的 ID、proposal、OFF action、bank threshold 均一致，frame 214/216/217
  的 candidate score 仍超容差。bank threshold 修复不是完整修复。
- **耗时瓶颈：** 当前 commit 的 20-record probe 为 `22.659 s`，857 次
  proposal、908 次 step；较大的 2400-record probe 为 `3771.883 s`，其中
  proposal/model evaluation 为实测主耗时。完整 v2 从 `04:15:18` 到退出
  `11:10:35 UTC`，即约 `6 h 55 min`。
- **决策：** video01 runtime gate `NO_GO`，candidate follow-up 等待语义
  修复；Full H8 继续 `PAUSED_NONCANONICAL`。不能把 v2 的 provenance PASS
  或小窗口 PASS 当作完整 runtime GO。

## 1. video1 六小时证据

证据报告：[`VIDEO01_SIX_HOUR_RUNTIME_EVIDENCE_20261007.json`](../reports/JEV_RNG_V4/VIDEO01_SIX_HOUR_RUNTIME_EVIDENCE_20261007.json)

| 项目 | 证据 |
|---|---|
| Builder PID | 12163 |
| 开始 | 2026-10-07 04:15:18 UTC |
| 完成 | 2026-10-07 11:10:35 UTC |
| 用时 | 06:55:17.54 |
| records | 8995 |
| JSONL 行数 | 8995 |
| records SHA-256 | `1da2ed9e7eec43c7cf139754d873f78a299c344ff74d65c6866b92f18f3da79` |
| source commit | `4108f18f5040432f68d55872e81e9f76e9acd08f` |
| checkpoint SHA-256 | `cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8` |

问题不在“有没有跑完”：它跑完了。问题在于后验语义门禁：

- provenance：PASS；checkpoint wrapper：PASS；
- runtime feature parity：FAIL，8995 条比较中最大绝对误差 `0.006004035472869873`，容差 `2e-5`；
- reactivation candidate parity：FAIL，225 个 mismatch；candidate ID/order 对，但 candidate scores 和 chosen proposal 不完全一致；
- runtime feature stability：FAIL，3 次重复均出现误差；
- closed-loop：因 pre-gate failure 未运行；
- 决策：`NO_GO_RUNTIME_SEMANTICS_GATE`，`FULL_H8_AUTHORIZED=false`。

因此不能因为“超过六小时并且 8995 条都写出来”就把它升级成 canonical 或论文结果。

本轮对 `1d2711e` segmented v4、`b20d7e2` 以及尚未合并修复的完整差异审计见
[`JEV_1D_V4_VS_B20D7_DIFF_AND_REMAINING_20261007.md`](JEV_1D_V4_VS_B20D7_DIFF_AND_REMAINING_20261007.md)。
该报告明确区分数据/语义 gate PASS、current-head controller quality NO-GO 和 Full H8
未授权状态。

另外，VisionTrack evaluator 顺序修复已经包含在 `1d2711e` 的 canonical source history
中（修复 commit `830c5cc`，回归证明 commit `623d16d`）；与之内容相同的独立候选分支
[`evaluator-order-fix-20261007`](https://github.com/LYQ1107/WWW/tree/jev/evaluator-order-fix-20261007)：
源码 commit `8dfc9e4`，回归测试 commit `c7ac234` 只是重复发布。`1d2711e` 集成分支
上的 3/3 回归也已通过；全部 22 个 test video、58,038 个 image 和 527,474 个
prediction rows 与 diagnostic remap 行级完全一致。修复尚未用于新的正式 inference，
但它不是当前 `1d` 的未合并问题。

## 2. 当前 segmented small gate

并发审计：[`SEGMENTED_CONCURRENCY_CHANGE_AUDIT_20261007.json`](../reports/JEV_RNG_V4/SEGMENTED_CONCURRENCY_CHANGE_AUDIT_20261007.json)。以下 MPS 数字是运行中的历史快照；最终 acceptance 状态以 `88/88 COMPLETE` 为准。

当前 runtime：`/home/liuyeqiang/WWW_jev_rng_v4_runtime/segmented_gate_20261007_v4`

- source commit 固定为 `1d2711e80ac5fa00806fd9eed30e90cd51df6b30`；
- 旧 driver PID 13608 在 MPS 切换前 graceful drain；恢复 driver PID 1819 后完成 acceptance，随后 graceful 停止；
- 7 张卡 `[2,3,5,6,7,8,9]`，当前 28 个 worker，即 4 workers/GPU，后端为 task-isolated NVIDIA MPS；
- 切换前已经完成的 chunk 被复用，partial 文件保留并作为审计/恢复证据，没有手工修改原始 queue；
- 最终 queue：`0 PENDING / 0 RUNNING / 88 COMPLETE`，总计 17491 条；
- video01：`2 pending / 11 running / 32 complete`；video06：`1 / 9 / 16`；video07：`2 / 8 / 7`；
- active chunk 名称无重复、每个 video 的 record ranges 唯一、没有 failure files；
- 12:04:59–12:05:14 UTC 抽样增加 250 条，观测约 16.59 records/s；12:12:31 UTC 最新快照为 13734 条。这些是运行性证据，不是稳定 ETA，也没有拿它宣称固定加速。

MPS 切换与等价性报告：[`SEGMENTED_MPS_SWITCH_AND_LATE_EQUIVALENCE_20261007.json`](../reports/JEV_RNG_V4/SEGMENTED_MPS_SWITCH_AND_LATE_EQUIVALENCE_20261007.json)

当前 28 个 worker 的 GPU 2/3/5/6/7/8/9 利用率约 26–35%，每卡约 4929–4937 MiB / 32508 MiB。作业继续保留，不做 kill -9、不迁移 video1。

## 3. 已通过的有限验证

- early bounded gate（frames 210–216）：70/70 records exact，candidate mismatch 0；
- late bounded gate（frames 881–890）：102/102 records exact，167 个 candidate events 比较，mismatch 0；
- basic tests：5/5 PASS；
- 这些是 segmented/chunk contract 的 bounded evidence，不等价于 full video1 runtime gate PASS。

## 4. 尚未完成与下一步

### segmented small gate closed-loop（两套 controller bundle，均为 screening only）

- video01/06/07 provenance、runtime parity、video01 candidate parity、video01 stability 均通过。
- Full JEV 在 video01 上为 HOTA `88.925`、AssA `90.022`、IDF1 `97.893`、MOTA `95.768`、IDSW `10`；相对同一 GMT OFF 的 ΔHOTA `+2.402`、ΔAssA `+4.873`、ΔIDF1 `+3.250`、ΔIDSW `-272`。
- Learnable Threshold 和 Generic MLP 分别严重退化到 HOTA `43.176` 和 `19.044`，因此结论是 JEV GO、Threshold/MLP WARNING；不能把三方法都写成成功。
- 详细报告：[`JEV_SEGMENTED_SMALL_GATE_CLOSED_LOOP_RESULT_20261007.md`](JEV_SEGMENTED_SMALL_GATE_CLOSED_LOOP_RESULT_20261007.md)。该结果不授权 canonical Full H8。

#### current-head bundle（最新 continuation，当前主判定）

- 使用最新 segmented video06/video07 数据：8496 records，dataset manifest SHA `3a0c5b05...f5df5`。
- GMT OFF：HOTA `86.523`、AssA `85.149`、IDF1 `94.642`、MOTA `89.579`、IDSW `282`。
- Full JEV：HOTA `85.072`、AssA `82.460`、IDF1 `91.975`、MOTA `95.586`、IDSW `18`，相对 OFF 的 ΔAssA `-2.689`、ΔIDSW `-264`。
- Threshold：HOTA `80.034`；Generic MLP：HOTA `68.217`；两者均低于 GMT OFF。
- 当前判定：`PILOT_NO_GO_CURRENT_HEAD_CONTROLLER_BUNDLE`；同一 bundle 的 repeatability 已通过（指标、动作计数及决策/预测 SHA 全部一致）。
- 机器可读报告：[`SEGMENTED_SMALL_GATE_CURRENT_HEAD_CLOSED_LOOP_VIDEO01_20261007.json`](../reports/JEV_RNG_V4/SEGMENTED_SMALL_GATE_CURRENT_HEAD_CLOSED_LOOP_VIDEO01_20261007.json)。

此前 JEV 正向结果使用另一 dataset/checkpoint bundle，只能作为独立 screening experiment，不能与本次 current-head 结果合并为重复实验。

1. 保留 small-gate 结果，完成 branch-consistent 的 learned-controller state/feature audit；当前 learned-vs-OFF 差值不能直接解释为 encoder mismatch。随后再决定是否把 video01 纳入新的 train/val 设计；不重复训练 video06/video07 旧三模型。
2. 完成 Full H8 authorization 要求的完整 chunk-equivalence、唯一冻结 source worktree 和 worker commit hard gate。
3. 只有上述审计完成且明确授权后，才重启 canonical Full H8；当前 Full H8 仍是 `PAUSED_NONCANONICAL`。

相关已推送分支：

- [audit-diagnostics-20261007](https://github.com/LYQ1107/WWW/tree/jev/audit-diagnostics-20261007)
- [video01-reactivation-box-parity-fix-20261007](https://github.com/LYQ1107/WWW/tree/jev/video01-reactivation-box-parity-fix-20261007)
- [video01-parity-fix-native-order-20261007](https://github.com/LYQ1107/WWW/tree/jev/video01-parity-fix-native-order-20261007)（候选修复，未合并）
- [segmented-anchor-geometry-integration-20261007](https://github.com/LYQ1107/WWW/tree/jev/segmented-anchor-geometry-integration-20261007)（`1d2711e` 集成候选，轻量回归通过，未做 full parity）
- [evaluator-order-fix-20261007](https://github.com/LYQ1107/WWW/tree/jev/evaluator-order-fix-20261007)（重复发布；修复已在 `1d2711e` 中）
