# JEV 分段 H8 构建、语义修复与三模型实验：独立复核报告

本次已完成 TRAIN 内 video01 / video06 / video07 的 17,491 条 H8 反事实记录构建、分段与串行等价检查、运行时验收，以及 Threshold / MLP / JEV 的训练、校准和独立 video01 闭环对比。JEV 优于两个学习对照，但其 AssA、HOTA、IDF1 仍低于 GMT OFF；本次质量结论为 **NO_GO**。这些结果是单种子、冻结感知缓存上的小规模筛查，不能作为官方 TEST 结果或 WWW 论文最终结论。

本报告与完整机器证据用于审查已经实际执行的工作。报告发布不会改变冻结实验代码，也不授权启动 Full24。

## 1. 固定版本和证据入口

- 修改基线：`4108f18f5040432f68d55872e81e9f76e9acd08f`。
- 实验冻结代码：`1d2711e80ac5fa00806fd9eed30e90cd51df6b30`。
- 审查分支：`jev/segmented-gate-review-20261007`，包含下面四个代码提交及本报告/证据提交。审查分支最终 HEAD 与实验冻结提交不同；产物 manifest 中的源版本始终是 `1d2711e`。
- 完整证据索引：[EVIDENCE_INDEX.json](../reports/JEV_SEGMENTED_GATE_20261007/EVIDENCE_INDEX.json)。索引记录原始路径、文件大小、SHA256、修改文件和冻结源码哈希。
- 执行汇总：[final_summary.json](../reports/JEV_SEGMENTED_GATE_20261007/evidence/final_summary.json)。
- 发布时重新校验：[PUBLICATION_INTEGRITY.json](../reports/JEV_SEGMENTED_GATE_20261007/PUBLICATION_INTEGRITY.json)、[publication_checks.json](../reports/JEV_SEGMENTED_GATE_20261007/publication_checks.json)。

原代码 worktree：`/data1/liuyeqiang/WWW_segmented_gate_20261007`。原实验目录：`/home/liuyeqiang/WWW_jev_rng_v4_runtime/segmented_gate_20261007_v4`。GitHub 包含报告原件、manifest、探针记录及运行辅助脚本；完整生产 JSONL、冻结 GMT checkpoint、感知缓存、GT 和策略权重仍在服务器。只有 GitHub 报告不足以重新训练或重跑完整跟踪，需要相同外部输入。

## 2. 使用哪些视频、如何划分

所有视频来自 VisionTrack TRAIN；划分单位为完整视频，没有按帧随机划分。

| 视频 | 用途 | MATCH | MEMORY | REACTIVATION | 总记录 |
| --- | --- | ---: | ---: | ---: | ---: |
| video07 / `00020court3` | 策略训练 | 1,703 | 1,631 | 0 | 3,334 |
| video06 / `00017court1` | 验证、温度校准 | 2,600 | 2,562 | 0 | 5,162 |
| video01 / `00002garden` | 独立闭环筛查 | 4,524 | 4,297 | 174 | 8,995 |
| 合计 | | 8,827 | 8,490 | 174 | 17,491 |

训练和验证序列均没有 REACTIVATION 标签。这是本次学习与泛化能力的重要限制，不能因为 video01 覆盖了 174 个 OFF 重关联事件，就认为模型已经充分学习该决策。

配置：`configs/VISION_test.yaml`；冻结 GMT：`model_20000.pth`；关联后端：`formal_gmt_transformer`；引擎：`cached_perception_mutable_association_v2`；H8；view_num=2；history_limit=80，实际窗口遵循现有生产顺序和窗口计算函数。

关键输入 SHA256（其余视频 trace、记录和模型哈希见 evidence index / manifests）：

| 输入 | SHA256 |
| --- | --- |
| GMT checkpoint | `cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8` |
| 感知 cache index | `48a51e1bfa89e9b360dc6b562801fc58092d83a4f60d9d04e4f6d92e4275ea1c` |
| TRAIN annotations | `9093c36204bae482c2f74997f531bfe2d8a46464f4541093703ef22db0068c85` |
| VISION_test config | `bdaeca71d875e8824c3eaf825967a7ba032514297642a6aabe7ad10a740be94a` |
| 原始 native video01 trace | `cb340485ecf348d2a63cacf69bcf9d3c3fd78a949a7d1cb8f5a9738d5c5a5e1b` |
| 映射后的 native video01 trace | `02773f217184600ce2eba7954e4a8fcd28c9070d8260e2a52ee07d670b0545b3` |

生产 video01 JSONL：`a174399cb979d5d1f9dc41366a4f07f6299c159c68e10e19222fedc656db7611`。发布前重新计算全部三个生产 JSONL 的哈希及问题计数，与 manifest 对照；另重新计算三模型原权重与校准权重共六个文件的哈希。

## 3. 修改了哪些代码

相对基线共修改 8 个文件，新增 680 行、删除 17 行。实际生产模型和冻结感知 checkpoint 没有重新训练。

| 提交 | 作用 |
| --- | --- |
| `7e32349` | 增加可观察、可恢复的分段 H8 构建和小视频三模型流水线；增加 builder 进度回调 |
| `c5043d8` | 向 LRU worker 显式传递完整视频 cache keys，修复缓存包装对象缺少 keys 接口的问题 |
| `6cc9402` | 对齐 raw ReID memory、joint stale bank、所有 unmatched query，并加入 native 检测行映射工具 |
| `1d2711e` | 对齐 native stale-bank 晋升时机、持久 bank 输入顺序及闭环调用时机 |

| 文件 | 具体修改与原因 |
| --- | --- |
| `reproduction_tools/run_segmented_small_gate.py`（新增） | 实现 `run/prepare/worker/merge/baseline/verify/aftercare`；按完整 frame/view 单元切段；保存 OFF/RNG 快照；锁保护队列；持久 worker 复用模型/cache/GT；分段写盘、恢复、合并及后续验收/训练/闭环 |
| `reproduction_tools/build_jev_counterfactual_v2.py` | 增加当前 key、候选动作和未来 key 的进度回调；bank query 包含全部 MATCH `START_NEW` unmatched 行，避免只使用带 REACTIVATION trace 的行 |
| `reproduction_tools/jev_counterfactual_v2.py` | formal 后端 memory 保存冻结 cache 的未归一化 ReID；最近 bank_size 个向量按 native 逆时间顺序累加；formal 路径不在每步末尾提前晋升 stale ID；保留 native `poss_ids.copy()` 遍历及 bank 字典插入顺序 |
| `reproduction_tools/jev_gmt_association_adapter.py` | 所有 stale ID 合成一个历史 `Instances`，使用 joint softmax；bank decoder 取 `outputs[0]`；保留历史 bank 输入顺序，输出候选仍按现有接口聚合 |
| `reproduction_tools/run_early_pilot_tracking.py` | bank proposal 保留持久 bank 顺序；仅存在 unmatched query 时才调用 bank candidates，保持 builder 和闭环语义一致 |
| `reproduction_tools/canonicalize_native_reactivation_rows.py`（新增） | 用 native bbox 在对应冻结 payload 中唯一匹配检测行，把 bank-local `detection_index` 映射为全局 cache row；不改 native 分数、候选或动作 |
| `reproduction_tools/test_jev_gmt_association_adapter.py` | 增加 raw magnitude、逆时间平均、joint bank softmax 和历史输入顺序回归检查 |
| `reproduction_tools/test_jev_reactivation_semantics.py` | 增加 proposal 保留 bank 插入顺序的检查 |

这些是语义修复与执行优化，不是新增 JEV 网络结构或候选条件化策略头。本报告只声明上述四个提交的工作；其他对话的修复分支没有自动合并进来。

### 3.1 为什么必须修复 memory 与 bank

原 mutable formal memory 保存 L2 归一化向量，而 native GMT 保存 raw 冻结 ReID。第一个重关联窗口中，native 分数约 `0.0041328063`，旧 replay 约 `0.07426155`。改变 memory 向量模长会改变 stale-bank transformer 输入，因此属于计算协议修复，需要重新生成相应反事实数据。

原 adapter 把各个 stale ID 作为独立历史 `Instances`，造成分别归一化；native old_reids 使用一个 joint `Instances`。新的回归测试使用两个历史 ID 和一个 unmatched dummy column，期望 joint score 为 `1/3`，可区分原来的分别 softmax。

后段 frame 881 附近的差异还涉及 Python set 构造和轨迹槽位 RNG 映射。ID 143 / 207 的输入顺序变化会影响映射；不能为了“排序稳定”改变 native 输入构造。修复保留 native 晋升时机和 bank 插入顺序，并通过包含这一窗口的串行/分段对照。

### 3.2 为什么要映射 native detection_index

native 重关联 trace 的 detection_index 是 bank-local unmatched 子集行，replay 使用全局 cache 行。直接比较会产生 native-only / replay-only key，即使候选分数完全相同。

本次映射 174 个 REACTIVATION 事件，其中 105 个行号发生变化；bbox 匹配容差 `2e-5`，要求唯一匹配，不依赖 GT，不使用模糊 fallback。原 trace 保留，映射文件及原/映射 SHA 均可审查：[native_trace_row_mapping.json](../reports/JEV_SEGMENTED_GATE_20261007/evidence/native_trace_row_mapping.json)。

## 4. 分段如何保证上下文、恢复和可观察性

仅划分输出 key range；每个 worker 仍读取完整 source trace、完整 actions/memory event map 和未来 cache keys。各段从按生产顺序生成的 OFF mutable state 开始，快照包含 association history、memory 及 Python trajectory RNG；H8 未来回放可以跨越段边界，不因输出切段而截断。

目标段大小约 200 条决策，实际按完整 frame/view 单元切分。三个视频共 88 段；video01 为 45 段。worker 使用锁保护的队列领取任务，模型/cache/GT 在进程内复用。

- 处理中持续写 `records.jsonl.partial`；约每 2 秒刷新具体 frame/view、key、action、future_key。
- 完成段经 fsync、原子改名后发布 manifest；完整段可经哈希/源绑定验证后复用。
- 中断段从该段快照重算，不能把未验收 partial 直接用于训练。
- 合并检查完整覆盖、无重复 semantic key、段哈希及源/输入绑定。
- 独立 driver lock 防止重复驱动同一输出目录。

辅助观察脚本：[check_progress.py](../reports/JEV_SEGMENTED_GATE_20261007/runtime_support/check_progress.py)。其原始用法是在原运行目录执行 `python check_progress.py --watch 10`；只读观察退出不会停止构建。进程存活或 GPU 利用率本身不能证明未卡死，应看处理时间戳和实际完成数是否持续变化。

## 5. GPU 与耗时口径

实际使用物理 GPU **2、3、5、6、7、8、9**。先运行 7 个 worker，再运行 21 个普通 CUDA worker；最后使用任务独立的 NVIDIA MPS，4 worker / GPU、总计 28 worker。切换时复用了同一冻结版本已完成的 25 段、4,969 条记录；中断的未完成段从快照重算。

[performance_samples.jsonl](../reports/JEV_SEGMENTED_GATE_20261007/evidence/performance_samples.jsonl) 显示三个视频合计实际完成速度：普通多进程阶段一次采样约 4.3 条/秒，MPS 阶段常见约 12–13 条/秒。各时间段处理的 chunk 难度不同，这不是严格控制变量的性能基准，不能作为论文加速倍数。

MPS 阶段于 `2026-10-07 12:00:07 UTC` 启动；合并 video01 manifest 于 `12:19:47 UTC` 生成。约 20 分钟对应剩余构建，不能表述为从零生成全部数据的耗时；此前预热、探针、开发、排错和后续训练/验收也需要时间。双方旧流程均已使用冻结感知缓存，主要新增提速来自分段并行和执行调度。

私有 MPS 服务结束后已关闭：[mps_cleanup.json](../reports/JEV_SEGMENTED_GATE_20261007/evidence/mps_cleanup.json)。历史启动信息、GPU UUID 与 wrapper 原件均已归档；未修改本机其他任务的 compute mode。

## 6. 验收结果及准确作用范围

| 检查 | 覆盖 | 结果 |
| --- | --- | --- |
| 串行 vs 分段 | frame 210–216：70 条；frame 881–890：102 条 | PASS；172 条 canonical record 完全一致，比较字段最大误差 0 |
| 普通 CUDA vs MPS | 上述两个窗口共 172 条 | PASS；canonical SHA 完全一致 |
| 生产段 vs 串行晚段探针 | 实际生产数据中对应 102 条 | PASS；无失配 |
| provenance/schema | 三视频共 17,491 条 | PASS；数量、问题覆盖、有限值、无重复及输入绑定检查 |
| builder vs OFF mutable runtime feature parity | video01 / 06 / 07 全记录 | PASS；这是共享 canonical builder 与 mutable runtime 的检查 |
| native vs mutable 重关联候选 | video01 全部 174 个 OFF 事件 | PASS；ID、顺序/规范化、分数、proposal、OFF action、事件覆盖均通过 |
| 固定 GPU 数值稳定性 | video01 全覆盖，3 次重复 | PASS |
| 三模型训练/校准、闭环执行 | 单种子，三个模型及 GMT OFF | 完成、执行 PASS；跟踪质量 NO_GO |

探针原记录也已归档，可独立逐条比较；不只是保存 PASS 字样。证据位于 `reports/JEV_SEGMENTED_GATE_20261007/evidence/probe/`、`late_probe/`、`mps_probe/`、`mps_late_probe/`。发布时对这四组原记录重新比较，核对 canonical SHA 与归档报告：[PROBE_RECHECK.json](../reports/JEV_SEGMENTED_GATE_20261007/PROBE_RECHECK.json)。

**全视频 feature parity 并非零误差。** video01 最大绝对误差为 `6.103515625e-5`，来自尺度敏感特征；已有冻结规则是 absolute `2e-5` 加仅对尺度敏感特征启用的 relative `4*EPS32 = 4.76837158203125e-7`。不能把它解释成所有字段都满足纯 absolute `2e-5`，也没有扩大 candidate score gate 的容差。完整 per-feature 和 tolerance 信息保存在 [parity1.json](../reports/JEV_SEGMENTED_GATE_20261007/evidence/reports/parity1.json) 和 [stability.json](../reports/JEV_SEGMENTED_GATE_20261007/evidence/reports/stability.json)。

发布前在新的干净审查 worktree 重新执行四个 main-based 合约检查及两个 pytest 文件，pytest **5 passed**。测试覆盖分支/RNG 隔离、snapshot/chunk equality、raw ReID、joint bank 及输入顺序；此次发布没有再次训练或重跑完整闭环。

## 7. 三模型协议与闭环结果

共同协议：seed `20261003`，20 epochs，AdamW，lr `0.001`，batch `128`；JEV hidden=128，MLP=139，Threshold=140；trainable params 分别为 34,080 / 34,163 / 33,987，相对差异均小于 0.3%。三模型统一在 video06 上进行温度校准，video01 不参与策略训练或温度拟合。

温度分别为 JEV `1.6119342427588557`、MLP `2.837101324209661`、Threshold `6.051448275069765`。具体参数、模型哈希和校准结果见归档 method manifests。

video01 冻结 detector/ReID payload + formal GMT association transformer + mutable branch-local state + 在线 typed controller 的闭环结果：

| 方法 | HOTA | AssA | IDF1 | IDSW ↓ | MOTA |
| --- | ---: | ---: | ---: | ---: | ---: |
| GMT OFF | 86.5230 | 85.1492 | 94.6425 | 282 | 89.5791 |
| Threshold | 80.0335 | 72.9168 | 83.9946 | 771 | 78.4528 |
| MLP | 68.2166 | 53.5144 | 72.8760 | 1,018 | 72.8328 |
| JEV | 85.0723 | 82.4599 | 91.9749 | 18 | 95.5859 |

JEV 相对 GMT OFF：HOTA −1.4507、AssA −2.6893、IDF1 −2.6676、IDSW −264、MOTA +6.0068。身份切换减少与关联质量下降同时存在，不能用 IDSW 或 MOTA 单独宣称整体更好。

[tracking.json](../reports/JEV_SEGMENTED_GATE_20261007/evidence/reports/tracking.json) 的 `status=PASS` 表示闭环执行成功；其 `pilot_verdict=PILOT_FAIL_RUNTIME_STATE_PARITY_OR_TRACKING` 是组合名称。这里 OFF runtime parity gate 已通过，失败的是跟踪质量。JEV 改变状态后的 feature 与原 OFF trace 不相同是闭环分岔，不应把 learned-policy 分支对 OFF trace 的大差值当作 OFF parity gate 失败。

另外，174 个 native/OFF REACTIVATION 事件全部执行 START_NEW，REACTIVATE_OLD=0。这与 JEV 闭环分支的 7 个 REACTIVATE_OLD 是不同轨迹；JEV action report 记录 `false_reactivation=2`，不能把 OFF 的 false_reactivation=0 当成 JEV 零误激活的结论。

## 8. 离线结果为何不足以支持结构优势

| 方法/参照 | video06 best-action accuracy | video06 utility |
| --- | ---: | ---: |
| Threshold | 0.936459 | 36.613425 |
| MLP | 0.984308 | 36.625630 |
| JEV | 0.999613 | 36.628971 |
| 按问题多数动作参照 | 0.999806 | 36.629504 |
| Oracle best utility | — | 36.629795 |

JEV utility 优于两个学习对照，但没有超过多数动作参照。video06 的 2,562 个 MEMORY 问题 **100% utility 并列**；全部验证记录 tie rate 约 51.86%。按 tie-aware accuracy 得到的高分不能直接证明学习到了有意义的 memory 决策。原始诊断：[training.json](../reports/JEV_SEGMENTED_GATE_20261007/evidence/reports/training.json)，其中 `meaningful_learned_policy_advantage=false`。

## 9. 已知限制、运行中断和未完成工作

### 9.1 原生 GTR 控制器特征向量尚未证明等价

本次 feature parity 是 builder 与 mutable runtime 的 canonical 64 维特征检查。candidate parity 是 native 与 mutable 的候选检查，两者都不等于完整 native GTRRCNN 控制器向量等价。

冻结代码 `gtr/modeling/meta_arch/gtr_rcnn.py` 的 native bank 调用传入 `view=0`、`frame_index=k`、`window_length=max(1,T)`；该 bank 调用内的 k/T 是局部虚拟 bank/current 序列坐标。mutable runtime 使用实际视频 frame/view 和生产窗口长度。native memory 初始 birth feature 与 mutable memory 的计数口径也需要单独核对。因此原生控制器特征接口必须另外对齐并验收，不能凭本次两个 PASS 宣称 full native 部署或端到端 native feature parity 已完成。

### 9.2 完整流程确实使用过外部续跑脚本

实际构建 driver 在合并后、后续验收阶段退出；根因没有确证，不归因为 OOM、其他对话或某个已知软件错误。已有的三视频 provenance/parity、candidate 和三次 stability PASS 被复用；随后通过外部 `continue_experiments.py` 完成三模型训练/校准和闭环。

第一次续跑因 tracked 生成报告 `STATE_FEATURE_PARITY.json` 改动被 source guard 拦下。续跑只排除这一已确认生成输出，保留其他 tracked 文件和所有组件/输入哈希检查；另一次因提取 aftercare tail 缺少 `records/methods` 路径变量失败，补齐后仅续跑闭环，未重复训练。两次已解决错误报告保留在证据包中。

原 worktree 最终有两个生成报告改动：`STATE_FEATURE_PARITY.json` 和 `RUNTIME_STATE_TRACKING.json`；发布使用新的干净 worktree，未把它们写回旧的通用报告位置。实验冻结后算法源码没有变化。归档的外部 wrapper/continuation 是历史运行原件、含固定路径和动态 aftercare tail 提取，不是可直接在任意机器运行的通用入口。

当前 `run_segmented_small_gate.py` 的完整段恢复已实现，但 aftercare 还未做通用幂等恢复；生成 tracked report 可能阻止整条 pipeline 直接重启。不得把本次“通过人工审查后的续跑成功”表述为所有阶段均已自动容错。

### 9.3 max_events 诊断不能代表同协议重现

此前 20 条 `max_events` probe 与完整构建记录的标签比较使用了不同未来 action/memory schedule：max_events 在构造 event maps 前截断事件。该比较不足以证明旧完整产物不可重现。

保留完整 trace、只通过 selected_key_range 限制输出的同协议 70 条检查，旧完整记录与相应旧语义探针的 outcomes、targets、best_actions、weights 全部一致：[full_trace_reward_audit.json](../reports/JEV_SEGMENTED_GATE_20261007/evidence/full_trace_reward_audit.json)。它是诊断证据，不是授权复用旧 labels；raw-memory / bank-order 等实际语义改变后，本次生产数据仍全部按新冻结版本构建。

### 9.4 下一步应当核对的事项

1. 对齐审查分支和其他修复分支：逐文件比较，不能直接混用旧 v2 JSONL 与新源码，或把某分支的 PASS 赋给另一个版本。
2. 独立复核 stale-bank promotion/input order 与 native 控制器 feature 接口，明确候选等价和完整向量等价的界限。
3. 分析 JEV IDSW 降低但 AssA/IDF1 下降的具体轨迹，以及错误合并、记忆跳过和数据分布的影响。
4. 检查 MEMORY tie 和训练/验证缺少 REACTIVATION 的数据问题，再决定下一轮训练协议及多种子验证。

当前 Full24 未授权、未启动；官方 TEST 未读取。本次报告不设置新的后台任务。

## 10. 复核命令与证据校验

取得审查分支后，在仓库根目录执行（校验本证据包的字节完整性）：

```bash
sha256sum -c reports/JEV_SEGMENTED_GATE_20261007/SHA256SUMS
git diff --stat 4108f18 1d2711e
git diff 4108f18 1d2711e -- reproduction_tools
```

在已配置 GMT 环境中运行轻量合约检查：

```bash
export PYTHONPATH=.:reproduction_tools:third_party/CenterNet2
python reproduction_tools/test_jev_counterfactual_v2.py
python reproduction_tools/test_jev_gmt_association_adapter.py
python reproduction_tools/test_jev_counterfactual_rng_isolation.py
python reproduction_tools/test_jev_reactivation_semantics.py
python -m pytest -q reproduction_tools/test_jev_intra_video_chunking.py reproduction_tools/test_jev_gmt_association_adapter.py
```

若要独立进行 GPU 分段等价重现，先检出实验提交到单独干净 worktree，并准备相同冻结 checkpoint/cache/annotations 以及映射后的 native trace。代码当前含原服务器绝对路径，跨机器运行需显式适配路径；适配后的源版本/哈希会变化，须作为新复核运行记录，不能覆盖旧 manifest。以下只给出两个有限窗口，不会启动整视频构建：

```bash
python reproduction_tools/run_segmented_small_gate.py verify --output /tmp/jev_review_210_216 --videos 1 --gpus 2 3 5 6 --frames 210 216 --chunk-records 20
python reproduction_tools/run_segmented_small_gate.py verify --output /tmp/jev_review_881_890 --videos 1 --gpus 2 3 5 6 --frames 881 890 --chunk-records 30
```

GPU 编号仅是原运行配置示例；复核者需选择本机可用 GPU。历史 MPS private daemon 已退出，不应直接运行归档 wrapper 并假定服务仍在线。SHA256 校验、合约测试和重现窗口分别验证文件完整性、逻辑不变量和运行等价；它们不替代对 utility 定义、数据覆盖和跟踪质量的科学审查。
