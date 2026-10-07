# `1d2711e` segmented v4 与 `b20d7e2` 差异及真实剩余问题

日期：2026-10-07 UTC
审计分支：`jev/audit-diagnostics-20261007`
本报告性质：差异审计与暂停决策，不是新的训练或 tracking 结果。

## 0. 本轮范围与硬停止

本轮已暂停旧 v2／`b20d7e2` 的重复计算。进程审计没有发现仍在运行的
WWW/JEV/video builder/Full-H8 任务；没有启动新的完整 video01 构建、三模型重训
或 Full H8。

本报告只读取既有产物和源码历史，并保存一个尚未合并的候选修复分支：

```text
jev/video01-parity-fix-native-order-20261007
5baa0dd Restore native stale-bank timing for parity fix
```

这个分支没有被合并到 `1d2711e`，也没有获得 Full H8 授权。

在此基础上又建立了一个以 `1d2711e` 为基线的源码集成候选：

```text
jev/segmented-anchor-geometry-integration-20261007
d4107b7a8425bf11134221a99a5909af3f55bf02
```

它只移植 stale-bank anchor geometry，并保留 1d 的 native order/timing/row-union；
GMT 环境下 adapter geometry、reactivation row semantics、counterfactual invariants
和 Python compile 均 PASS。它尚未做 full video01 candidate/runtime parity，因此仍是
候选修复，不是 canonical 结果。

## 1. 结论先行

| 项目 | 结论 | 说明 |
|---|---|---|
| `1d2711e` segmented data gate | **PASS** | 88/88 segments，17,491 条记录；early 70/70、late 102/102 分段等价；MPS 172/172 等价 |
| provenance / feature / candidate / stability | **PASS** | video01/06/07 provenance PASS；video01 feature parity、174/174 candidate parity、3 次 stability PASS |
| offline 三模型 | **pipeline PASS，研究优势 NO-GO** | JEV 只略高于另外两个 learned controller，仍低于 majority reference；数据存在明显 tie/label degeneracy |
| current-head closed loop | **NO-GO** | JEV 的 HOTA/AssA/IDF1 仍低于 GMT OFF；Threshold/MLP 明显退化 |
| learned-controller runtime feature parity | **FAIL / 未解决** | learned controller 没有 canonical feature records，最大绝对误差约 1,783–2,273 |
| candidate-conditioned reactivation supervision | **未解决** | reactivation 的 GT candidate recall 为 0；没有 per-candidate long-horizon return |
| Full H8 / 24-video canonical build | **NOT AUTHORIZED** | official TEST 未读，Full H8 保持暂停、非 canonical |

因此，`1d2711e` 的含义是：**数据构建和 GMT OFF 语义回放已经过 gate，但新增 controller
还没有通过可以支撑最终论文结论的闭环质量 gate。** 不能因为 88 个 chunk 完成，
就把 Full H8 或任何三模型结果升级为正式结果。

## 2. `segmented_gate_20261007_v4` 完整验收

固定源 commit：

```text
1d2711e80ac5fa00806fd9eed30e90cd51df6b30
```

| video | records | MATCH | MEMORY | REACTIVATION | records SHA-256 |
|---|---:|---:|---:|---:|---|
| video01 | 8,995 | 4,524 | 4,297 | 174 | `a174399cb979d5d1f9dc41366a4f07f6299c159c68e10e19222fedc656db7611` |
| video06 | 5,162 | 2,600 | 2,562 | 0 | `4ffcb359673d020410b249f9a05dcb88fce06122b4a02ddd2e6260e6e9f99dd0` |
| video07 | 3,334 | 1,703 | 1,631 | 0 | `6abd8770b23427aaee3982695c93f27f4fceabaf0cf91328f2f0465dc4e79ae2` |
| **total** | **17,491** | 8,827 | 8,490 | 174 | — |

固定验收结果：

- `final_summary.status = EXPERIMENTS_COMPLETE`；`segments = 88`。
- segmented 单 worker/chunk contract：`70/70` early exact、`102/102` late exact。
- MPS 前后 equivalence：`172/172` records exact。
- video01 native/replay candidate parity：`174/174`，candidate IDs、scores、chosen proposal IDs、legacy OFF actions、bank threshold 均在冻结容差内一致。
- feature parity：8,995/8,995 finite，最大绝对误差 `6.103515625e-05`，超过容差数为 0。
- stability：3 次重复 PASS，输入和 records SHA 一致。
- `code_changes_after_source_freeze = false`；`full24_authorized = false`；`official_test_read = false`。

这里的 `PASS` 是数据/回放/稳定性验收，不代表 learned controller 的科学效果已经 PASS。

## 3. 三模型 offline 结果

配置：H=8、state dim=64、single seed `20261003`、20 epochs、batch 128、AdamW、
learning rate `0.001`、temperature scaling 只在 policy-val 使用。dataset SHA：
`3a0c5b05956d62ea09837678261559ac94d65ba77ba8eb16710c183f724f5df5`。

| Method | Val NLL | Best-action Accuracy | Brier | ECE | Validation Utility |
|---|---:|---:|---:|---:|---:|
| Learnable Threshold | 0.894494 | 0.936459 | 0.016869 | 0.488739 | 36.613425 |
| Generic MLP | 0.880879 | 0.984308 | 0.009480 | 0.510136 | 36.625630 |
| Full JEV | **0.878797** | **0.999613** | **0.007632** | 0.540902 | **36.628971** |
| Majority reference | — | 0.999806 | — | — | **36.629504** |
| Uniform legal-action reference | — | 0.678871 | — | — | 36.416634 |

解释：JEV 在三个 learned controller 中最高，但只比 MLP 高 `0.003342` utility，
仍低于 majority reference `0.000533`。全体 best-action tie rate 为 `51.8597%`；
其中 MEMORY_DECISION 为 `2,562/2,562` 全 tie，只有 `2,485/5,162` 条记录是 unique best。
所以当前 offline 结果不能证明存在 meaningful learned policy advantage。

## 4. current-head 三路闭环结果

以下是 `CORRECTED_V4_SMALL_H8_CLOSED_LOOP_DIAGNOSTIC_NOT_OFFICIAL_TEST`，同一
video01、同一 detector/perception、同一 GMT OFF baseline。Delta 均相对 GMT OFF。

| Method | HOTA | AssA | IDF1 | MOTA | IDSW | Frag | ΔHOTA | ΔAssA | ΔIDF1 | ΔMOTA | ΔIDSW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GMT OFF | 86.523 | 85.149 | 94.642 | 89.579 | 282 | 6 | 0 | 0 | 0 | 0 | 0 |
| Learnable Threshold | 80.034 | 72.917 | 83.995 | 78.453 | 771 | 6 | -6.490 | -12.232 | -10.648 | -11.126 | +489 |
| Generic MLP | 68.217 | 53.514 | 72.876 | 72.833 | 1,018 | 6 | -18.306 | -31.635 | -21.766 | -16.746 | +736 |
| Full JEV | 85.072 | 82.460 | 91.975 | 95.586 | **18** | 6 | -1.451 | -2.689 | -2.668 | **+6.007** | **-264** |

JEV 的 IDSW/MOTA 有改善，但核心 association 指标 HOTA、AssA、IDF1 仍低于 OFF，
因此最终 quality gate 是 `NO_GO_ASSA_BELOW_GMT_OFF`。Threshold/MLP 的闭环明显退化。

### 为什么不能引用旧的“JEV 正向结果”作为当前结论

旧目录 `closed_loop_video01_corrected_v4/PILOT_TRACKING_THREE_WAY.json` 属于另一套
screening-only dataset/checkpoint bundle，曾报告 JEV HOTA `88.925`、AssA `90.022`、
IDF1 `97.893`。它不是当前 `1d2711e` current-head bundle 的重复实验，不能与本节结果
拼接，也不能用来绕过当前 runtime feature mismatch。

## 5. `1d2711e` 与 `b20d7e2` 的源码差异

共同祖先是：

```text
4108f18f5040432f68d55872e81e9f76e9acd08f
```

两条分支不是线性关系：`1d2711e` 继承了 `6cc9402` 的 native raw-memory / joint stale
bank 语义，并加入 segmented small-gate pipeline；`b20d7e2` 则直接从 `4108f18` 增加
reactivation box-parity 修复。因此不能把 b20d7 的全部 diff 当成对 1d 的安全补丁。

### `1d2711e` 已经包含的关键内容

- native raw ReID memory 与 joint stale-bank proposal；
- native first decoder output、trajectory input order 和 reverse raw accumulation；
- unmatched `START_NEW` rows 并入 stale-bank transformer proposal；
- `canonicalize_native_reactivation_rows.py`、`run_segmented_small_gate.py` 和
  88-segment/固定 source commit 的验收流程；
- video01 candidate identity/order、score、proposal 与 stability 的完整可审计报告。

### `b20d7e2` 的真实新增修复

它修复了一个确实遗漏的 native geometry 语义：stale ID 进入 `old_reids` 时，保留原始
anchor box 和 image size，只替换 ReID feature 为 bank average；adapter 也改为把整个
stale bank 作为一个 `Instances` 对象送入 transformer。这会影响 positional input，
不是可忽略的格式调整。相关 adapter tests 已加入 b20 分支。

### 为什么不能直接 cherry-pick b20d7

以原始 `b20d7e2` 为基准的全量 mapped candidate diagnostic 虽然有 `174/174` 事件覆盖、
没有 missing key，但有 10 个 score/proposal mismatch，集中在 frames
`881, 885, 890, 894, 897, 903, 905, 913, 924, 930`。除 anchor geometry 外，原 b20
还带入了以下与 1d 不一致的行为：

- `possible_memory_ids` 被 `sorted(...)` 遍历，而 native 依赖 insertion order；
- stale bank 用 `torch.stack(...).mean(dim=0)`，而 native 是 reverse-order float32 raw sum；
- `step()` 对非 formal adapter 也 eager promote stale bank，改变 bank timing/order；
- b20 的 builder 去掉了 1d 中把 unmatched `START_NEW` rows 并入 proposal 的逻辑；
- b20 分支不包含 1d 的 segmented scheduler 和 row canonicalizer。

因此正确方向是“把 b20 的 anchor geometry 修复移植到 1d 的 native order/timing 语义中”，
而不是整体覆盖 1d。

## 6. 尚未合并的修复

| 修复 | 当前状态 | 是否已证明完整 |
|---|---|---|
| stale-bank 原始 anchor box/image size 保留 | 已在 `b20d7e2`，候选分支继承 | 仅 bounded frame-220 5/5；尚未在 1d full 174 events 上完成最终整合验收 |
| 1d 基线上的 anchor geometry 集成 | 候选 commit `d4107b7a8425bf11134221a99a5909af3f55bf02` | 轻量回归 PASS；full candidate/runtime parity 未运行 |
| native stale-bank insertion order、raw accumulation、formal timing 恢复 | 候选 commit `5baa0dd` | unit tests PASS；bounded frame-220 5/5 PASS；按硬停止要求未重跑 full parity |
| 1d unmatched `START_NEW` row union 恢复 | 同在 `5baa0dd` | 代码已恢复，尚未做 full 1d integration gate |
| b20 geometry 修复整合进 `1d2711e` | **未合并** | 需要新的固定 source commit、full candidate parity、runtime feature parity 和 stability |
| b20 adapter tests 与 1d segmented tests 合并 | **未合并** | 需要在统一 source tree 中重新跑测试和 provenance |

`5baa0dd` 只是在独立候选分支保存修复，不能被解释为 `1d2711e` 已通过新的 full gate。

## 7. 真正剩余的问题（按阻塞程度）

### P0：learned controller 的 runtime feature interface mismatch

当前-head tracking 报告中 learned controller 的 `canonical_feature_records` 全部为 0；
mutable runtime feature 与训练/canonical feature 不是同一接口：

| Method | max feature error | feature parity records | runtime records | wrong commit | no candidate | 其它 |
|---|---:|---:|---:|---:|---:|---|
| Threshold | 1,783.024 | 8,200 | 8,404 | 594 | 644 | — |
| MLP | 1,967.994 | 8,378 | 8,574 | 1,047 | 474 | memory contamination 245 |
| JEV | 2,272.641 | 8,809 | 9,030 | 16 | 18 | false reactivation 2 |

所以当前三路闭环只能证明“某个 mutable runtime 接口下的诊断行为”，不能证明论文中
要比较的 controller 在与训练数据同语义的 GMT runtime 上有效。必须先统一 feature
construction、memory/commit semantics 和 reactivation branch semantics。

### P0：reactivation candidate supervision 不可用

candidate identity audit 显示：MATCH candidate set recall 约 `0.997944`，但
REACTIVATION candidate set recall 为 `0`，0 个事件包含当前 GT candidate；同时 native
trace 没有 per-candidate long-horizon return。当前 labels 不能支撑 candidate-conditioned
JEV 的可靠监督，也不能用 GT-only offline mapping 冒充训练监督。

### P0：current-head association quality 未达标

JEV 虽减少 IDSW，但 AssA/HOTA/IDF1 相对 GMT OFF 仍为负；MLP/Threshold 更差。
在修复 runtime parity 之前，不应根据这些数值选择最终 controller，也不应扩大到 Full H8。

### P1：目标函数和数据分布存在 tie degeneracy

MEMORY_DECISION 全部 tie，overall tie rate 超过 51%；JEV 的 utility 仍不超过 majority
reference。需要明确 tie policy、informative weighting 和 candidate-level target 后，
才能判断模型是否真的学到策略，而不是复现多数动作。

### P1：统计稳健性和官方评测尚未完成

本轮按临时政策只跑 seed `20261003`；`20261004/20261005` 尚未补。official TEST 未读，
24-video canonical Full H8 也未授权/未构建。

## 8. 恢复工作前必须满足的 gate（本轮不执行）

1. 在统一 frozen source tree 中保留 1d 的 native order/timing、segmented tools 和
   canonicalizer，并移植 b20 的 anchor geometry。
2. 先做 bounded early/late candidate parity；再做完整 video01 174-event parity、
   feature parity、stability 和 provenance。任一失败都不进入重训。
3. 只有 learned controller 的 train/runtime feature construction 完全一致后，才重新
   评估三模型闭环；如果仍低于 OFF，结论就是 NO-GO，不启动 Full H8。
4. 通过新的 video01 gate 后，创建唯一 `CANONICAL_H8_COMMIT` frozen worktree，才有资格
   重新讨论 24-video Full H8。

以上步骤在本轮按用户指令保持暂停。

## 9. 机器可读证据与 SHA-256

| 产物 | SHA-256 |
|---|---|
| `final_summary.json` | `14257792121931cb3f9a62ef3b9a4b3532f1def0036501cc986d3c7fe9f5d863` |
| `reports/training.json` | `1157acfd166d0beae2c38bde6273386f8284a5778eb7ffff1ff1067d23f68cbe` |
| `reports/tracking.json` | `d3232d01ab4d93d39cd3555b037bda9d3ed596b09f9dd17d3f5f96abc66acee7` |
| `reports/candidate_parity.json` | `dc51514b3569e416de377a91ab39391a79e780687a9a4cbd9abb9de98034d6f7` |
| `reports/parity1.json` | `cac020f41d6c53ebe6d0f62e0d106f761c11fd0ed9ec1d818e2634f486bed703` |
| `reports/stability.json` | `a8bb8da864fe1e6395665607c9507a3c1ef565f9745c18628a3bc9e23da8cc29` |
| `probe/equivalence.json` | `31c1808f06f462b3cac155c65c8b6839156b2ba438f8d1f42cc251f91d011a6c` |
| `late_probe/equivalence.json` | `76d10273dba22246cc28359cfc9f2f7a9f843ccf44d113a6b32fbd3e9d531a80` |
