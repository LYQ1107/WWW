# JEV Phase VI 最终实验与代码复核报告

**结论：NO-GO。** 当前证据不足以支持完整 WWW Identity Lifecycle Decision Engine，也不足以授权 Full24、百万记录构建、官方 TEST 或大规模 sweep。已执行用户第二、第三停止条件：移除 learned MEMORY，关闭 canonical learned REACTIVATION。保留冻结 B2 和所有正、负结果。

研究分支：`jev/www-jev-lifecycle-phase6-20261008`。独立 worktree 从 Phase V `a37083dac0a23eb3846cb252eeb975982aaaabc9` 创建；Phase V HEAD 与工作区未改动。实验日期为 2026-10-07 UTC，目录名 20261008 沿用用户指定名称。

## 核心问题的答案

| 问题 | 本阶段证据支持的答案 |
| --- | --- |
| MATCH 是 structured association 还是 fancy gating？ | 两者都有贡献。native 全局重匹配在 video02 有独立因果收益；原 development 三动作优于 binary 的门槛失败，不能宣称三动作普遍必要。 |
| MEMORY 能否被救活？ | 固定 quality/representation 规则有效；20 个单 WRITE/READ 配对没有可训练监督，learned M6 未获救。 |
| REACTIVATION 能否被救活？ | 修正零样本与 feature contract 后获得有效监督，但完整 val23 的 HOTA/AssA 下降；按第三停止条件关闭。 |
| Unified JEV 超过 MATCH-only 吗？ | 未建立。前置门槛失败，因此未训练共享 Unified，也未补造 Full 行。 |
| WWW 完整三阶段故事成立吗？ | 不成立。当前保留 MATCH 研究及 MEMORY/REACT 诊断，不能包装成已成功的全生命周期方法。 |
| 应授权 Full24 吗？ | 否；官方 TEST 也未使用、未授权。 |

## 冻结 anchor 与 native 语义修正

B2 checkpoint SHA256 永久保持：

`f2aa3dd2b564d90bfbfb62dc0518b0b2107a931ed7134f8c0d19f5d152b94ed7`

原 research resolver 的 video01 结果逐字节复现：HOTA 88.4716、AssA 89.0855、IDSW 87，prediction SHA256 为 `cac367c0c2c8b44660bda8643e5c5003fcdc4e06069fe147c5da10e369f3ebde`。GMT 同样精确复现；二者均保留原有路径和名称。

源码审计发现 research resolver 在 constrained Hungarian 后直接提交，legacy threshold 只影响 MEMORY eligibility；真正 `GTRRCNN._apply_jev_match_decisions` 还对 ACCEPT/REASSOCIATE 行进行冻结 B2 的二次 learned binary 校验。11 个原事件首轮 64D features 和 actions 全部相同，提交结果却 11/11 不同，MEMORY eligibility 1/11 不同。仅比较 feature 或旧预测 SHA 不能证明 native state transition 等价。

新增 opt-in `NativeMatchResolver` 调用实际 native hook，提交与 MEMORY eligibility 在这 11 个事件全部 PASS；机制开关重构后重新审计仍 PASS。修正后 native video01 HOTA 88.3495、AssA 88.7962、IDSW 99，prediction SHA256 为 `2aeb3e9b7d2df5e047286f7c1a441208466596b6afd52e24af386caee9866a41`。这是单独标记的补充结果，不覆盖旧 positive anchor。生产端 relative-REACT hook 未实现，**完整 native lifecycle contract 未建立**。

见 [native transition audit](../reports/JEV_PHASE6/NATIVE_TRANSITION_CONTRACT_AUDIT.json)。

## 外部源码审计

修改实现前审计了用户指定的 20 个仓库，固定 commit，进入 52 个实际源码文件及相应函数。逐仓库 SHA、文件、函数、启发与不能照搬的部分见 [external code audit](JEV_PHASE6_EXTERNAL_CODE_AUDIT.md) 和 [source manifest](../reports/JEV_PHASE6/EXTERNAL_SOURCE_MANIFEST.json)。

关键实现启发：association 与 birth/lost/refind 必须具有明确 state transition；memory quality、bounded gallery 与 representation 应独立验证；BoT-SORT alpha 是旧向量权重，QDTrack momentum 是新向量权重；离线图方法的未来信息不能进入在线 JEV。动作数多本身不构成结构化推理的新颖性证据。

## Fancy-gating 与原 11 次 REASSOCIATE

训练固定 train07 / val06、seed 20261003、20 epochs、AdamW 0.001、batch128、last checkpoint、仅验证集温度校准。video01 只用于既有 development/diagnostic，不用于选择新模型。G2 为真实 `score > τ(state)` 的 34,036 参数门限网络；G3 保留 34,080 参数 JEV 并重新训练 binary action，使用原 counterfactual utility soft targets，未随意将 REASSOCIATE 标签折叠。

| 条件 | 含义 | HOTA | AssA | IDSW |
| --- | --- | --- | --- | --- |
| G0 | GMT OFF | 86.5230 | 85.1492 | 282 |
| G1 | score-only scalar threshold | 88.6723 | 89.4247 | 59 |
| G2 | 64D dynamic threshold | 82.7997 | 77.9864 | 654 |
| G3 | binary JEV retrained | 88.9540 | 90.0206 | 29 |
| G4a | frozen B2 REASSOCIATE→NEW | 88.5062 | 89.1111 | 85 |
| G4b | frozen B2 REASSOCIATE→ACCEPT | 88.5049 | 89.1086 | 85 |
| G5 | frozen B2 original research resolver | 88.4716 | 89.0855 | 87 |

原 `STRUCTURED_MATCH_CLAIM=FAIL`：G5 未超过 G3，G4 移除第三动作未伤害 development 指标。由于 G2 明显更差，用户第一条“binary≈dynamic≈three-action”精确等价停止条件不成立；但 Unified 的 structured-MATCH 前置条件仍失败。后续只进行有界组件监督和新 heldout 证明。

原 11 个事件做了 CONTROL / FORCE_ACCEPT / FORCE_NEW 共 33 次独立 live-policy forks，并报告未来 1/2/4/8/16/32。H32 保留 REASSOCIATE 相对两种替换各有 2 次收益，相对 ACCEPT 有 3 次损失、相对 NEW 有 4 次损失；仅 3 个当前 detection 可与 GT 匹配，观察性 N01=2、N10=1。重叠时间窗不独立，8 个 GT-unmatched 不自动算错。**该事件审计属于原 research 语义**，不能直接当成修正 native 的因果结论。

见 [gating audit](../reports/JEV_PHASE6/FANCY_GATING_AUDIT.json)、[11-event audit](../reports/JEV_PHASE6/REASSOCIATE_EVENT_AUDIT.json)。

## Native 全局关联的独立机制证据

对原预注册 video02/03/05 增加固定 checkpoint 的 9 次 native 机制对照，分别在结果前记录协议。G4a/G4b 强制移除 REASSOCIATE，会同时移除全局求解与其触发的二次校验。为避免混淆，VALIDATION_ONLY 保留原 latent REASSOCIATE 触发与实际二次 learned 校验，但使用原 Hungarian pairs，隔离全局重新分配的作用。每个对照之后的全部未来状态重新在线计算，不套用缓存未来动作。

下表均为 **ΔHOTA / ΔAssA**：

| 对照 | 02：B2−对照 | 03：B2−对照 | 05：B2−对照 | pooled：B2−对照 |
| --- | --- | --- | --- | --- |
| G4a | +1.2850 / +2.4659 | -0.0103 / -0.0056 | -0.0245 / -0.0490 | +0.2974 / +0.5920 |
| G4b | +1.6422 / +3.1680 | +0.1925 / +0.3871 | -0.0245 / -0.0490 | +0.5077 / +1.0034 |
| VALIDATION_ONLY | +1.2873 / +2.4665 | +0.0130 / +0.0405 | -0.0245 / -0.0490 | +0.3127 / +0.6212 |

保留二次校验、只去掉全局重分配仍损失 video02 约 1.2873 HOTA / 2.4665 AssA，因此“全部增益都只是 gating”过强。该收益在 video03 很小、video05 略负；pooled 因果优势不等于每个序列均获益，也不推翻原 development gate FAIL。见 [mechanism audit](../reports/JEV_PHASE6/NATIVE_HELDOUT_MECHANISM_AUDIT.json)。

## Canonical lifecycle 数据与真实闭环

全部有界 lifecycle 数据重新建立在 native MATCH 上；先前 legacy prefix 明确排除训练。video24 有 1,058、video23 有 445 个真实 REACT candidate events，使用固定 seed 的无 GT reservoir 抽取 32/16 个事件；各抽 10 个 MEMORY events（8 个按首次实际 READ 选择，2 个晚期 unread 控制）。合计 68 个事件 × CONTROL/两个动作 = **204 次 live-policy forks**。

分段 journal 每 64 个 committed payload 写盘；通过真实提交及 RNG 重建 factual prefixes，避免第二遍 GMT forward。恢复 fingerprint 校验涵盖身份、active/stale IDs、ordered bank、assignments、relative metadata 和 RNG，明确排除诊断 counters 与 immutable perception，不能称为所有 mutable 字段无差异。全部 CONTROL 与 native baseline 对应未来一致；未来干预后不使用 factual action cache。

REACT 使用 13 个相对 online fields，MEMORY 8 个 quality/history fields；绝对帧号和 virtual frame 不进入输入。assigned candidate score 单独保留，因为 Hungarian proposal 不必是该行 top1。GT 仅离线使用 cache0→annotation1、精确 image IDs、IoU≥0.5 one-to-one，为 utility 和评测服务。

在 1,503 个 REACT events 中，满足条件的多个 stale IDs 与“assigned 错而存在有效正确 alternate”均为 0，没有添加无证据的第三个 stale action。因果 horizon camera-end 另行审计，零个同帧不同 camera-end 的不等窗口，零个有效标签受影响。unknown-target 的延迟/恢复项在训练前被离线 censor，所有这类样本原本 weight=0；保存的 training source SHA 与最终 label 文件精确一致。

见 [canonical audit](../reports/JEV_PHASE6/REACTIVATION_CANONICAL_DATA_AUDIT.json)、[action space](../reports/JEV_PHASE6/REACTIVATION_ACTION_SPACE_AUDIT.json)、[horizon audit](../reports/JEV_PHASE6/REACT_CAUSAL_HORIZON_VIEW_AUDIT.json)。

## MEMORY：真实 READ 仍无单次 WRITE 监督，固定规则有收益

比较当前 WRITE/SKIP，一直运行到 WRITE 分支的实际首次 stale-bank READ 后 16 帧，两个分支使用共同结束点，同时报告 READ+8/+16。未读样本 weight=0，unknown identity weight=0，utility ties weight=0；GT 不用于挑选事件。utility 包含正确/错误持续时长、switch、fragmentation、contamination、target query rank 和 latency。

20 个样本中：14 个 known-target READ 配对 utility tie，2 个 READ 配对目标未知，4 个未读；全部训练权重为 0。16 个实际 READ 配对的 bank prototype SHA 均因 WRITE/SKIP 而改变，但没有改变声明的下游 utility。该 read-enriched 小样本不是无偏 tie prevalence；单次最后 WRITE、bounded horizon 与目标查询暴露仍有限。**M6 未训练、没有虚构 checkpoint，第二停止条件已执行。**

另做同一 native B2 MATCH 的完整 train24/val23 M0–M5 共 12 次结果：

| 条件 | video24 HOTA / AssA | video23 HOTA / AssA |
| --- | --- | --- |
| M0 | 60.7789 / 56.0941 | 61.0002 / 56.7954 |
| M1 | 61.3575 / 57.1053 | 61.5460 / 57.8897 |
| M2 | 60.7789 / 56.0941 | 61.0002 / 56.7954 |
| M3 | 61.6164 / 57.5760 | 61.3133 / 57.4200 |
| M4 | 61.1787 / 56.6983 | 61.5409 / 57.8737 |
| M5 | 60.6729 / 55.8338 | 61.8871 / 58.5193 |

M0 always-write 的 native latest10 raw-vector mean；M1 confidence≥0.6；M2 bounded latest10，预测 SHA 与 M0 在两个完整视频完全一致；M3 EMA old-weight0.9；M4 old-weight=`0.9+0.1*(1-trust)`、`trust=clip((conf-0.4)/0.6)`；M5 new-weight0.8 momentum。使用原始 GMT ReID 向量，复用更新公式，不声称完整复现外部归一化 pipeline。

M1 两序列 HOTA/AssA 都提高；M3 在24较强、M5在23较强，不能看完结果再宣称存在通用最佳规则。这说明 memory quality/representation 有价值，但没有建立 learned JEV MEMORY 优于 M0–M5。见 [event protocol](../reports/JEV_PHASE6/MEMORY_EVENT_DRIVEN_PROTOCOL.json)、[baseline comparison](../reports/JEV_PHASE6/MEMORY_BASELINE_COMPARISON.json)。

## REACTIVATION：有效监督出现，但验证失败，按要求关闭

32/16 个 counterfactual labels 中，有效监督为 train24 **16** / val23 **9**。固定同样 20 epochs、seed、optimizer、last checkpoint 和 val-only temperature，禁止 video01/heldout 选模型。第20轮训练 loss 0.5777、val weighted accuracy 0.4444、NLL 0.7120，仅作诊断；最终看完整在线 tracking：

| 序列 | native B2 HOTA / AssA | 加 canonical REACT | ΔHOTA / ΔAssA |
| --- | --- | --- | --- |
| video24 (TRAIN) | 60.7789 / 56.0941 | 61.4369 / 57.2813 | +0.6580 / +1.1873 |
| video23 (VAL) | 61.0002 / 56.7954 | 60.6248 / 56.1067 | -0.3754 / -0.6887 |

第三停止条件触发：canonical REACT 虽在训练视频提高，却在验证视频同时损害 HOTA 和 AssA。因此 learned REACT 关闭，不继续增加训练预算或状态增强以强行保留故事。小数据、单 seed 失败不证明该概念永久无效。

Typed prototype 为 **39,712 参数**，满足 <50K；本次 standalone 诊断实际另用冻结 **34,080 参数 B2 MATCH**，总 runtime **73,792 参数**。它没有完成共享核心的联合训练，不能作为 shared Unified 已成功的证据。MATCH、MEMORY 前置门槛不齐，Full、learned MEMORY、clean+state-corruption、新的公平 Generic MLP/Full ablation 均未运行。历史 Phase V MLP collapse 单独保留，不拿来填新的公平对照表。

记录完整性说明：`STANDALONE_CLOSED_LOOP_STOP_PROTOCOL.json` 写于指标计算结束之后、下一次工具读取之前，已更正为 **既有用户第三停止条件的事后文档**，不能称为结果前预注册；停止规则本身来自用户最初 goal，早于所有实验。未运行的 augmentation 方案同样不声称预注册结果或鲁棒性收益。

见 [typed architecture](../reports/JEV_PHASE6/TYPED_JEV_ARCHITECTURE.json)、[augmentation scope](../reports/JEV_PHASE6/CLOSED_LOOP_STATE_AUGMENTATION.json)、[unified ablation](../reports/JEV_PHASE6/UNIFIED_LIFECYCLE_ABLATION.json)。

## 三个预注册完整 TRAIN controller-heldouts

视频 02/03/05 在首个结果前固定，没有按表现挑视频、重新调模型或换 seed。它们此前未用于 Phase V controller 参数选择；GMT backbone 可能已见过，因此称为 **TRAIN controller-heldouts**。video01 绝不是独立 test；不涉及官方 TEST。

保留最初 15 次 G0/G1/G2/G3/G5 research 结果，并另列 3 次修正 native B2。pooled 从三序列重新运行真实 TrackEval combine，**不是 HOTA/AssA 算术平均**。

| 条件 | video02 HOTA / AssA | video03 HOTA / AssA | video05 HOTA / AssA | 真实 pooled HOTA / AssA |
| --- | --- | --- | --- | --- |
| G0 | 75.6009 / 74.0486 | 77.7373 / 78.1753 | 86.9565 / 86.9183 | 78.3941 / 78.4797 |
| G1 | 74.6106 / 72.1599 | 77.6040 / 77.9446 | 87.5683 / 88.1418 | 78.1529 / 78.0311 |
| G2 | 62.3892 / 50.6659 | 34.8295 / 15.8205 | 87.2057 / 87.4146 | 51.1004 / 33.2964 |
| G3 | 74.3650 / 71.6891 | 77.0732 / 76.8733 | 87.5893 / 88.1837 | 77.7601 / 77.2411 |
| G5 | 73.0162 / 69.1477 | 77.7849 / 78.2923 | 87.5379 / 88.0810 | 77.8917 / 77.5155 |
| G5_NATIVE_TRANSITION | 75.9252 / 74.6785 | 77.6421 / 78.0110 | 87.5502 / 88.1058 | 78.4883 / 78.6813 |

Native B2 在02/05的两项主指标均正，03略负；pooled ΔHOTA **+0.0942**、ΔAssA **+0.2017**，IDSW 560→333。确实达到“multiple positive”字面条件，但未满足事先声明的“三个都正” gate；不能事后放宽该 gate。G2 video03 闭环 collapse 也完整保留，不能因 IDSW 下降就判方法更优。

所有结果使用一致继承的 TRAIN GT duplicate override，属于研究诊断分数，不是官方 TEST 排名。见 [heldout protocol](../reports/JEV_PHASE6/HELDOUT_PROTOCOL.json)、[full results](../reports/JEV_PHASE6/HELDOUT_RESULTS.json)。

## 每阶段 WHAT DID WE LEARN

| Phase | 完成范围与学到的内容 |
| --- | --- |
| 0 | 独立 branch/worktree；B2 与 Phase V 原证据冻结，研究允许失败。 |
| 1 | 20 个源码仓库支持不同生命周期语义和时间尺度；动作数与复杂模型本身不是贡献。 |
| 2 | 原 development 上 binary 优于三动作，不能宣称第三动作普遍必要；dynamic threshold 不是所有状态下的等价替代。 |
| 3 | 原11事件收益与损失并存；native commit parity 比 first-feature parity 更强。新固定权重机制实验进一步识别 video02 全局关联贡献。 |
| 4 | 实际 READ 确实发生，prototype 也改变，但20单 WRITE边际监督均无信息；固定 memory质量规则仍有价值。 |
| 5 | canonical相对特征和实际 candidate采样修复零监督问题；完整val23失败表明覆盖率本身不能救活 learned REACT。 |
| 6 | action-space诊断没有支持更多 stale动作；按原停止条件未做 corruption增强，鲁棒性问题仍未解决。 |
| 7 | typed相对state设计能在<50K内实现；standalone训练不是共享reasoner联合训练的证据。 |
| 8 | structured MATCH 与 MEMORY前置gate失败，Unified训练未获资格；不输出假 checkpoint。 |
| 9 | 有界组件比较完成；公平Full/新GenericMLP联合ablation未运行，Full> MATCH未建立。 |
| 10 | 固定三个完整TRAIN controller-heldouts，native收益2/3且pooled小幅正，原all-three gate仍失败。 |
| 11 | 多个必要门槛失败，最终NO-GO；Full24、官方TEST、大规模构建与sweep继续禁止。 |

## 修改的代码与核验

| 文件 | 作用 |
| --- | --- |
| `gtr/modeling/jev_lifecycle.py` | typed adapters、relative-state contract 与共享核心原型，未开启生产Full。 |
| `reproduction_tools/jev_phase6_native_match.py` | opt-in实际native二次校验、真实提交及MEMORY eligibility；validation-only机制开关。 |
| `reproduction_tools/jev_phase6_rollouts.py` | 分段committed journal、prefix重建、native canonical采样、实际READ、live counterfactual分支。 |
| `reproduction_tools/jev_phase6_label_contract.py` | 离线unknown-target censor与标签contract，GT不进入运行输入。 |
| `reproduction_tools/jev_phase6_fast_match.py` | 缓存固定question/action算子，保持浮点运算与softmax语义；拒绝不精确JIT候选。 |
| `reproduction_tools/run_jev_phase6_*.py` | gating、rare-event、native tracking/removal、memory baseline、head完整视频、显存规划队列。 |
| `reproduction_tools/fit_jev_phase6_head.py` | 完整native来源/有信息监督资格核验、固定预算standalone训练和校准。 |
| `reproduction_tools/audit_jev_phase6_*.py` | native提交、canonical覆盖与action-space审计。 |
| `reproduction_tools/summarize_jev_phase6_*.py` | 原heldout及native机制真实pooled总结，不均值替代。 |
| `reproduction_tools/archive_jev_phase6.py` | 确定性gzip/分片与source/archive SHA绑定，保留原始预测、标签、checkpoint、评测。 |
| `reproduction_tools/finalize_jev_phase6.py` | 强制验证完成证据、停止条件、B2/PhaseV，输出最终gate判断。 |
| `reproduction_tools/write_jev_phase6_final_report.py` | 从已完成JSON生成本复核报告。 |
| `reproduction_tools/verify_jev_phase6_publication.py` | 校验归档SHA、解压/分片重建、projection完整输入及训练标签绑定。 |

相关gate/utility contract的5项单测已通过；native11事件修正后的commit/MEMORY parity通过；完整video01的4,523条MATCH logits逐位相同且native prediction SHA完全一致；compact诊断context同样通过完整回归；journal identity/RNG/metadata重建通过。曾有JIT trace产生2.98e−8差异，候选明确拒绝，未部署。日志瘦身仅去除无用diagnostic lifetime maps，不修改实际model features或候选决策。没有重训或覆盖B2。

## 复核入口与原始证据边界

所有必需JSON位于 [reports/JEV_PHASE6](../reports/JEV_PHASE6)，最终gate见 [FINAL_PHASE6_GO_NO_GO.json](../reports/JEV_PHASE6/FINAL_PHASE6_GO_NO_GO.json)。各实验 `evidence/.../ARCHIVE_MANIFEST.json` 声明源文件路径、source SHA256、归档路径、archive SHA256和字节数；`.gz` 解压恢复原字节，`.partNNNNN.gz` 按编号解压后拼接恢复整个JSON或JSONL。原大JSON按字节分片保留空格与换行，降低单次Git传输量。

| 证据目录 | 内容 |
| --- | --- |
| `evidence/gating_training`, `evidence/gating_tracking` | G1–G3 checkpoint/训练及G0–G5development原始预测、决策、输入、评测。 |
| `evidence/reassociate` | 原11事件的33个分支、baseline和评测。 |
| `evidence/heldouts`, `evidence/native_b2`, `evidence/heldout_pooled` | 原15次heldout、修正native3次及真实pooled。 |
| `evidence/lifecycle_native/video24`, `video23` | native baseline、canonical输入、所有68个事件及204个分支的labels/预测/决策/READ/评测。 |
| `evidence/memory_baselines_native` | 完整M0–M5比较；M0源预测复用上行native baseline，SHA绑定。 |
| `evidence/standalone_heads_native`, `evidence/standalone_closed_loop_native` | canonicalREACT checkpoint/训练与完整24/23预测、typed输入、评测；MEMORY insufficient-data记录。 |
| `evidence/native_removal`, `evidence/native_control_pooled` | 9次native机制对照和3条件真实pooled。 |
| `evidence/execution_manifests` | 实际queue命令、最终无pending/running状态和有意替换supervisor的历史说明。 |

大型原始prefix snapshots/committed journal二进制保留在本机runtime，GitHub保留其SHA manifest与重建代码，**未上传这些大型运行状态**。GT formatter副本、prepared中间文件、不可变perception NPY未上传，可从声明数据重建；原预测与TrackEval结果独立上传。G2 video03 collapse产生>512MB的巨大diagnostic candidate数组，归档明确标记为projection，保留每行完整64D控制器输入、合法/所选动作、proposal、RNG metadata，只截断候选ID/score数组到8并保留原count；它不是完整candidate数组的lossless副本。全量源文件与其SHA仍在runtime，不将projection误称为全部原日志。

运行环境：`/home/liuyeqiang/anaconda3/envs/GMT/bin/python`。原runtime：`/home/liuyeqiang/WWW_jev_phase6_runtime/20261008`。审阅者可在取得声明TRAIN数据与perception cache后运行各脚本的`--help`与对应已保存queue/task配置；无需官方TEST，也不应启动Full24。

```bash
git checkout jev/www-jev-lifecycle-phase6-20261008
PY=/home/liuyeqiang/anaconda3/envs/GMT/bin/python
$PY reproduction_tools/finalize_jev_phase6.py
$PY reproduction_tools/write_jev_phase6_final_report.py
```

上述finalizer需要原runtime或按source manifest恢复到约定路径；查看已归档JSON/预测/checkpoint不需要启动GPU实验。后续若研究memory bank级干预、更有目标暴露的监督或真实state-corruption，是新实验协议，不属于本次已经完成并停止的GO任务。
