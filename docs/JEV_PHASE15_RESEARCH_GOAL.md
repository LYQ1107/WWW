# FINAL GOAL — WWW / JEV PHASE XV
## Persistent Identity Commitment: Reliable GTA-Free Visual JEV for Online Multi-Camera Multi-Object Tracking

### 目标：彻底解决身份碎片反复切换、UNKNOWN 历史误关联和在线状态失稳，使 JEV 真正成为可靠的 MCMOT 身份关联模型

**Repository**

https://github.com/LYQ1107/WWW

**Frozen starting commit**

`0dc9607192c22bb4487b77b41de20f188caf7f1a`

**New research branch**

`jev/www-jev-phase15-persistent-identity-commitment-20261010`

**Reference reports**

- `docs/JEV_PHASE13_FINAL_RESEARCH_REPORT.md`
- `docs/JEV_PHASE14_FINAL_RESEARCH_REPORT.md`
- `reports/JEV_PHASE14/FINAL_GO_NO_GO.json`
- `reports/JEV_PHASE14/ERROR_ATTRIBUTION.json`
- `reports/JEV_PHASE14/FALSE_MERGE_SPLIT_DIAGNOSTICS.json`
- `reports/JEV_PHASE14/UNKNOWN_CHOICE_FORENSICS.json`
- `reports/JEV_PHASE14/ON_POLICY_STABILITY.json`
- `reports/JEV_PHASE14/PAIRED_NATIVE_FUTURE_RESULTS.json`

---

# 0. FINAL GOAL：必须交付一个真正稳定工作的 JEV-MCMOT

我授权你在合法科研数据范围内自主开展必要的诊断、算法修改、模型训练、原生反事实实验、公平基线比较、性能优化和多轮迭代。

**不要把本任务理解为运行一轮实验、写一份报告、得到 NO_GO 后就自动宣告科研目标完成。**

需要尽最大合理努力解决问题，而不是简单复述历史失败原因。

核心目标：

构建一个真正能够在严格在线条件下，利用视觉证据和长期 Global Identity Memory，直接完成身份选择、身份延续和受约束身份恢复的 JEV-MCMOT 模型。

必须：

1. 保留已经验证的 GTA-free 身份关联架构。
2. 继续由 JEV 直接预测真实 Global Identity 选项。
3. 真正解决同一 GT 的多个纯净预测身份碎片之间频繁跳转的问题。
4. 对 UNKNOWN 和污染历史建立可验证的风险判断。
5. 增加真实的 Persistent Identity Commitment 决策，而非固定的持续匹配阈值。
6. 防止过度坚持旧 ID 导致错误合并。
7. 使用真实在线状态训练。
8. 严格对照普通 Fixed Question、Set Transformer 和原始 GMT。
9. 使用完整视频闭环和官方评测验证实际跟踪质量。
10. 报告完整实时性能与泛化限制。

**只有所有必要的科研门槛均通过时，才能将 Scientific Goal 标记为 GO。**

若经过有界、证据驱动的迭代仍无法解决，应保存全部结果，明确当前科学硬阻塞，标记 `SCIENTIFIC_NO_GO` 或 `BLOCKED`，而不是制造成功。

任务执行完成与科研目标成功是两个不同状态。

---

# 1. 冻结此前事实：不得继续混淆身份识别与身份延续

Phase XIV 已经证明：

- Multi-question JEV 可以学习 WHO。
- 可以学习 Identity Availability。
- 可以学习 Observed History Trust。
- 原生 GTA-free 关联与长期状态能够正常运行。
- 多问题在跨相机 IDF1 上有一定正向信号。
- 但 HOTA 和 IDSW 稳定性未通过研究 Gate。

正式结果：

| Method | HOTA | IDSW | CVIDF1 |
|---|---:|---:|---:|
| Phase XIV Fixed Question | 73.067 | 171.000 | 86.405 |
| Phase XIV Multi Question | 73.505 | 391.667 | 88.546 |
| Phase XIV Set Transformer | 72.073 | 207.000 | 85.398 |
| Phase XIII Fixed Question | 75.637 | 105.000 | 91.889 |
| Original GMT separate system | 78.110 | 204.000 | 94.814 |

这些不同阶段的方法可能有不同的训练目标，不能全部作为同输入同监督的因果对照。

Phase XIV Multi 正式三种子的 IDSW：

- Seed 20261008：181。
- Seed 20261009：824。
- Seed 20261010：170。

其中 seed20261009：

- video17：329 次 IDSW。
- video18：432 次 IDSW。
- video19：63 次 IDSW。

该种子中：

- PURE_FRAGMENT_SWITCH_OR_RECOVERY：270。
- UNKNOWN_OR_CONTAMINATED_EXISTING_CHANGE：519。
- 其他类别：35。

当前首要任务不是继续优化总体 CE，而是定位并减少前两类问题。

---

# 2. 核心科研假设

当前模型的主要问题：

**GT Identity Recognition 与 Persistent Online Track Identity Commitment 不一致。**

例如同一个真实身份 A 可能对应三个历史预测 ID：

`17, 28, 43`

三个历史 ID 均可能是纯净的。

当前 WHO 多正例监督可能允许：

`P(ID17) + P(ID28) + P(ID43)`

共同构成正确身份概率质量。

但真实在线跟踪需要在能够安全维持身份时，持续使用同一个已经提交的 ID。

否则：

`17 → 28 → 17 → 43`

WHO 可以全部认为是正确的，但 CLEAR IDSW 依然增加。

因此必须把两种问题分开：

**Identity Recognition**

“这个当前观测与哪些历史身份的视觉和时空证据兼容？”

**Identity Commitment**

“在已有在线身份状态和历史承诺下，这一次应该继续使用哪个具体 Global Track ID，或者确实需要切换？”

这两个问题既相关，又不等价。

需要通过真实实验检验：

H1：多正例身份学习缺少稳定 Track ID 的偏好，是 PURE_FRAGMENT_SWITCH 的重要原因。

H2：UNKNOWN 或污染历史的候选被模型反复选择，是身份状态失稳的重要原因。

H3：通过可学习的 Commitment 决策，而非简单固定阈值，可以兼顾身份连续性和错误身份纠正。

H4：Commitment 与 WHO/Availability/Trust 共享 Jev 决策表示，是否具有普通网络不能在相同条件下完全解释的独立收益。

所有假设必须允许被证伪。

---

# 3. P0 — 保护 Phase XIV 资产与建立独立工作树

先核验：

`0dc9607192c22bb4487b77b41de20f188caf7f1a`

创建新 research worktree。

不允许覆盖：

- Phase V–XIV 全部 checkpoint。
- 原有323份 checkpoint 归档。
- Phase XIII/XIV 的原始输出、失败日志、manifest。
- 完整原生前缀。
- 已完成的 MATLAB 结果。
- 原有冻结评测协议。

读取并核验所需 SHA。

本轮新增：

`reports/JEV_PHASE15/`

`docs/JEV_PHASE15_*`

`gtr/modeling/jev_phase15/`

`reproduction_tools/jev_phase15_*`

所有代码更改不得无声影响原始 GMT OFF 和历史 Phase XIV 回放。

输出：

`PHASE14_FROZEN_EVIDENCE.json`

`SOURCE_AND_CHECKPOINT_MANIFEST.json`

---

# 4. P1 — 首先复现824次 IDSW 最差种子的真实失稳

**在完成该任务前，不启动正式神经网络长训练。**

优先：

`multi_question / seed20261009 / video17`

`multi_question / seed20261009 / video18`

还需对照：

`fixed_question / seed20261009`

`set_transformer / seed20261009`

及其他两个 Multi seeds。

## 4.1 按原生前缀定位 IDSW

使用已有的：

`reproduction_tools/jev_phase14_forensics.py`

`reproduction_tools/jev_phase14_native_risk.py`

`reports/JEV_PHASE14/ERROR_ATTRIBUTION.json`

并读取对应服务器原生日志。

逐个恢复：

- Frame。
- Camera。
- Detection。
- GT（仅离线分析）。
- 当前预测 Track ID。
- 上一个同 GT 观测的预测 Track ID。
- 当前合法候选 Global IDs。
- 当前 WHO logits。
- Availability logit。
- Trust logits。
- 具体 Option scores。
- 候选 Gallery visual tokens。
- Identity history。
- Native assignment。
- Bank state。
- 最终已提交 ID。
- 接下来的连续 ID 变化。

不能只统计总 IDSW。

## 4.2 对每次身份切换进行精确分类

至少区分：

A. PURE_FRAGMENT_HOP。

同一个 GT 在不同纯净预测 Track IDs 之间跳转。

B. POLLUTED_HISTORY_SWITCH。

从已有污染历史中选择其他预测 ID。

C. CORRECTIVE_SWITCH。

此前预测身份确实有错误，切换具有合理纠错作用。

D. WRONG_EXISTING_MERGE。

错误把另一真实人物提交到当前 ID。

E. FALSE_BIRTH。

原始身份已经存在，但重新创建新的预测 ID。

F. RECOVERY_SWITCH。

恢复旧 ID 造成身份变化。

G. AMBIGUOUS/UNKNOWN。

无法基于可靠 GT 和历史认证切换性质。

## 4.3 不能将所有 IDSW 都视为错误决策

必须区分：

- CLEAR 评价的真实 IDSW。
- 纯碎片 hop。
- 当前可证明的错误合并。
- 为纠正已经污染的身份而进行的合理切换。
- GT 不可认证的变化。

尤其需要验证：

**当模型错误地坚持此前 ID 时，可能会不会让另一个真实身份长期遭到错误合并？**

因此不能将“保持上一次 ID”设为永远正确。

输出：

`PHASE15_SWITCH_EVENT_LEDGER.json`

`PHASE15_SEED20261009_FAILURE_ATLAS.json`

`PHASE15_CORRECTIVE_VS_HARMFUL_SWITCH.json`

---

# 5. P2 — 先做无训练的原生反事实实验，验证 Commitment 是否有因果价值

从 TRAIN 和已经保存的开发诊断前缀中，选择可靠的身份切换事件。

TRAIN 可以用于训练监督。

开发视频的 GT 只能用于独立诊断和评测，不得进入训练或在线动作输入。

对同一个真实 native prefix，在合法条件下比较：

1. `KEEP_PREVIOUS_COMMITTED_ID`
2. `SELECT_JEV_BEST_ID`
3. `SELECT_ANOTHER_CERTIFIED_CANDIDATE`（仅离线研究分支）
4. `DEFER`
5. 原生固定/保守 fallback

其中第三种使用离线身份认证的动作，只能作为研究期上界或反事实干预，不允许进入实际部署。

必须保证所有动作确实合法：

- 同相机 ID 唯一。
- 不改变其他摄像头的真实历史。
- 不使用未来帧选择动作。
- 不修改历史已输出预测。
- 不制造候选。
- 不重写 GT。
- 不跳过真实 Memory/Bank 状态更新。

## 5.1 真实未来分支

使用相同 RNG 和相同真实起点。

测：

H8、H16、H32。

必要时增加 H64。

每条分支保存：

- 真实 committed identity。
- IDSW。
- Identity continuation。
- False merge。
- False split。
- False birth。
- Gallery contamination。
- Cross-camera identity continuity。
- 实际效用差异。

后续策略必须明确标记：

`π_fixed`

`π_full`

`π_multi`

或其他冻结 policy。

不能把一个后续 policy 的动作价值冒充另一个 policy 的价值。

## 5.2 关键 Gate

必须回答：

**如果在 824 次 IDSW 的关键前缀处，仅仅改变具体身份承诺动作，是否可以减少后续身份切换，而且不增加错误合并？**

如果答案为 YES：

优先研究 Commitment Head。

如果 NO：

进一步研究身份候选生成、历史污染和恢复机制。

如果许多切换根本不存在安全可选的已有 ID：

不允许简单训练一个 KEEP_HEAD 去制造虚假的连续性。

输出：

`COMMITMENT_ORACLE_UPPER_BOUND.json`

`COMMITMENT_NATIVE_CAUSAL_BRANCHES.json`

`COMMITMENT_FEASIBILITY_GO_NO_GO.json`

---

# 6. P3 — 新增真正的 Persistent Identity Commitment State

这是本次最重要的结构设计。

不要重新训练一个完全独立的 MOT Backbone。

保留当前：

`GlobalIdentityJev`

`ReliableIdentityPolicy`

`IdentityMemory`

`CachedIdentityMemory`

`NativeDirectExecutor`

在其上增加持久身份承诺状态：

`CommitmentState`

## 6.1 CommitmentState 必须存储真实过去信息

至少包含：

- 当前正在使用的 Global Track ID。
- 每个摄像头的最近已提交身份观测。
- 最近提交时间。
- 最近检测框及外观摘要。
- 轨迹连续性。
- 同一相机中的潜在身份冲突。
- Global Identity Memory。
- 近期跨摄像头关联记录。
- 当前候选身份的观测支持。
- 历史身份纯净度预测。
- 当前身份已经持续使用的时间。
- 已知的不一致或断裂事件。

注意当前新检测本身没有预先分配的真实身份。

如果需要知道它可能延续哪个上一帧检测，必须通过真实、因果合法的运动/视觉对应关系建立候选，而不能直接从 GT 读取上一帧人物 ID。

对于模糊的短期对应，应明确保留多个合法假设或 UNKNOWN，而不是伪造唯一前驱。

## 6.2 State 必须具备真实持久性

新增：

`commitment_state.py`

要求：

- 每帧持续更新。
- 跨 camera payload 共享。
- 支持 snapshot。
- 支持 serialize/restore。
- 支持同视角唯一身份约束。
- 支持跨视角共享 Global ID。
- 支持 stale identity。
- 支持原生 ID recycling。
- 支持中断续跑。
- 不保存未来信息。
- 不保存 GT。

不能仅在 forward 里创建一个临时 Question embedding 后就称为 Persistent Identity Memory。

---

# 7. P4 — 新增真正的 Jev Commitment Question

新的 Jev 不只有 WHO、Availability 和 Trust。

增加一个核心问题：

**Q4 — COMMIT**

“What specific persistent Global Track ID should be committed now, given the previous identity commitments and current visual evidence?”

它必须有真正独立的输入和动作语义。

## 7.1 Question 输入

Q4 使用：

- Current visual state。
- Existing Global Identity candidates。
- Previous committed identity evidence。
- Camera-specific continuity。
- Candidate conflicts。
- Temporal gaps。
- Trust/contamination evidence。
- WHO scores。
- 当前合法 action set。

不能只通过固定 task embedding 表示。

Q4 应通过动态 QuestionReader 读取多个真实候选和历史承诺 Tokens。

## 7.2 每个候选都要有 Commitment 特征

对当前检测 i、候选身份 j：

`CommitmentOption(i,j)`

应包含：

- 是否与可靠前驱 ID 相同。
- 此前是否在当前摄像头连续使用。
- 是否在另一个摄像头已建立全局身份。
- 时间间隔。
- 视觉连续性。
- 历史身份污染风险。
- 候选竞争冲突。
- 更换该 ID 的潜在风险。
- 上一轮合法分配中的身份一致性。

这些必须由真实过去状态计算。

不能使用 `candidate ID == GT identity` 作为输入。

## 7.3 输出可学习动作价值

例如：

`P(CONTINUE_ID_j | state)`

`P(SWITCH_TO_ID_j | state)`

`P(DEFER | state)`

这些只是类型化决策语义，不要求对每个候选使用完全独立的新分类器。

推荐直接为合法具体动作学习 Commitment-adjusted value：

`V_commit(state, question, action)`

可以以现有 WHO 分数为基础，加入 Commitment 对每一个具体候选的差异化调整。

不能只给全部 existing identities 增加同一个标量。

必须真正改变同一 GT 多个候选碎片之间的决策排序。

## 7.4 不得退化成固定 Hysteresis

禁止只实施：

`if previous_id exists: always use previous_id`

或：

`if score_gap < threshold: keep previous`

这类简单静态规则。

它们可以作为强 baseline，但不能作为新 JEV 的核心创新。

新模型必须能够区分：

- 应继续使用纯净旧 ID。
- 旧 ID 已污染，应合理切换。
- 旧 ID 对应前驱关系不可信。
- 跨摄像头存在更可靠全局 ID。
- 当前确实没有可靠身份可用。

---

# 8. P5 — 正确处理“同一个 GT 的多个纯净预测碎片”

这是本次损失函数修改的核心。

当前多个同 GT 的纯净预测身份都可能获得 WHO 正例。

这部分可以保留，因为它们的确可能在视觉身份层面属于同一人物。

**但是必须增加具体 Track ID 的持续性监督。**

## 8.1 分离两级监督

第一级：

`WHO / IDENTITY_COMPATIBILITY`

同一 GT 的全部可靠纯净候选可以是正例。

第二级：

`COMMITMENT / ID_CONTINUATION`

在当前历史状态下，哪些具体 ID 应被优先延续？

当一个此前正确、仍然合法、没有污染且存在可靠当前连续性证据的 ID 可用时，它应获得独立的 Commitment 优先监督。

其他同 GT 的纯净碎片不能简单与它等价。

## 8.2 Commitment Label 必须可认证

基于 TRAIN 中：

- 当前 GT。
- 历史 predicted track 对 GT 的可靠对应。
- 上次实际已提交 ID。
- 当前摄像头时序。
- 真实候选合法性。
- 历史污染程度。

仅在证据足以判断安全延续或必要切换时，生成监督标签。

如果多个候选无法可靠区分，Commitment label 必须为 UNKNOWN。

不能在所有情况下强制“最早创建的 ID”正确。

不能简单以 ID 整数小、轨迹长度最长作为永远正确的真值。

## 8.3 必须包含应切换的监督

例如：

此前 Track 17 已经受到另一个人的错误观测污染，Track 28 是当前真实人物可靠的新持续轨迹。

模型应该能够学习：

`SWITCH 17 → 28`

而不是为了减少 IDSW 继续使用污染 ID 17。

因此新的 Commitment loss 需要包含：

- Safe continuation。
- Necessary correction。
- Unsafe switch。
- Ambiguous choice。

不能只对切换动作一律加惩罚。

---

# 9. P6 — 设计真实的 Commitment-Aware Joint Association Loss

至少保留：

`L_WHO`

`L_Availability`

`L_Trust`

`L_Assignment`

新增：

`L_Commitment`

以及可辨识的：

`L_UnsafeSwitchRisk`

建议从以下形式开始研究：

`L_total = L_WHO + λa L_Availability + λt L_Trust + λc L_Commitment + λj L_JointAssignment + λr L_Risk`

所有 λ 在正式验证之前冻结。

不能通过查看开发视频的 HOTA 随意调参。

## 9.1 Commitment Loss

对于可靠认证的延续事件：

优先选择已正确使用、仍合法并且未污染的具体 Track ID。

对于已确认应纠错的事件：

不惩罚必要的合法切换。

对于 ambiguous 事件：

不制造 Commitment 强监督。

## 9.2 Joint Assignment Loss

必须在真实的整个检测集合上学习。

不能只进行逐行独立分类。

保持：

- 相机内一对一。
- 跨相机共享 Global ID。
- 独立 DEFER terminal。
- 真实 Stale Bank 约束。
- 原生提交动作。

## 9.3 UNKNOWN 安全性

在 UNKNOWN 监督下：

- UNKNOWN 不能被当成可靠负例。
- UNKNOWN 也不能自动被视为可靠正例。
- 未认证候选仍属于合法推理选项。
- 模型应根据真实可观测证据估计选择风险。
- 选择性决策的覆盖率与风险必须分别测量。

必要时学习 Candidate Reliability posterior。

但不得根据开发集 GT 临时禁止 UNKNOWN 候选参与运行。

## 9.4 关键对照

同一数据、同一算力下训练：

A. 原 Multi Question + 原 Loss。

B. 原 Multi Question + 普通 ID continuity penalty。

C. 原 Multi Question + Learnable Commitment。

D. Fixed Question + 同样 Commitment 监督。

E. Set Transformer + 同样 Commitment 监督。

F. Full JEV + WHO/Availability/Trust/Commitment。

这样才能证明提升究竟来自：

- 延续损失。
- 新增过去状态。
- 普通网络也能学会的正则化。
- Jev 特有的动态问题条件化。

---

# 10. P7 — 身份碎片管理与可选的安全合并机制

不能只靠训练分数而完全忽略 Global Identity Memory 中已经存在的历史碎片。

但也不能为了减少 IDSW 随意合并 Global IDs。

新增研究一个独立模块：

`Persistent Identity Fragment Controller`

用于处理：

- 同一潜在身份的多个历史 Track IDs。
- 当前重复候选。
- 跨摄像头已有身份。
- 片段间视觉一致性。
- 时空不兼容证据。
- Identity history contamination。

## 10.1 优先考虑延续，不立即做全量 Merge

第一阶段只学习：

- 继续已有可靠 ID。
- 避免没有理由地跳向另一个碎片。
- 在错误旧身份确需纠正时进行切换。

先评估它本身的价值。

## 10.2 可选的在线 Alias/Consolidation

只有在真实证据满足事前冻结条件时，才允许研究身份片段合并。

必须保证：

- 不是用 GT 合并。
- 不是使用未来帧合并。
- 不重写此前已输出的视频身份。
- 不把两个同时存在且可能属于不同人的轨迹强制合并。
- 不造成同摄像头身份冲突。
- 能够正确处理旧 Gallery/Bank/引用。
- 具备冲突回滚或 fail-closed 机制。
- 每次状态变化可审计。

在线 Alias 不得变成事后 GT-based ID renumbering。

必须单独消融：

`No Alias`

`Conservative Online Alias`

`Learned Commitment only`

`Commitment + Qualified Alias`

若安全证据不足，Alias 模块可以保持关闭。

---

# 11. P8 — 专门针对 UNKNOWN / 污染历史建立运行时风险决策

当前主要失败之一是：

历史 Track ID 已污染，但模型仍可反复选择它。

利用 TRAIN predicted states 建立：

`Candidate Reliability Dataset`

同时保存：

- 当前真实可观测视觉特征。
- 过去的 Gallery 一致性。
- Candidate history length。
- 跨摄像头视觉支持。
- 当前视图与历史外观差异。
- 历史特征分布。
- 近期身份切换。
- 当前候选竞争。
- 离线认证的纯净/污染标签。

GT 标签只能用于训练监督。

部署时必须仅使用这些可观测特征预测可靠性。

## 11.1 训练 Candidate Reliability

输出：

`P(candidate history reliable | causal state)`

需要特别检验：

- 已知纯净历史。
- 已知污染历史。
- 难以认证的历史。
- 正确候选存在但污染候选得分更高。
- 两个纯净同 GT 碎片共同存在。
- 不同 GT 但视觉极其相似的候选。

## 11.2 风险不等于二分类

要区分：

- History purity。
- Current identity compatibility。
- Commitment safety。
- Action uncertainty。

一个纯净历史仍可能属于错误人物。

因此 `Trust=high` 不能作为“选择它”的充分条件。

一个已污染历史也不意味着所有该身份的观测必然无效。

必须建立分离的风险语义。

---

# 12. P9 — 先完成有界 Pilot，再进行正式训练

不要重复 Phase XIV 那种直接大量运行全模型矩阵的过程。

第一步：

冻结真实 TRAIN Commitment dataset。

第二步：

冻结少量真正有代表性的失败前缀。

第三步：

执行1000–2000 updates的 Pilot。

每个 Pilot 至少验证：

- 可学习性。
- Identity continuation。
- Necessary correction。
- False merge。
- False split。
- UNKNOWN selection。
- Candidate permutation。
- Cross-camera ID reuse。
- Native State parity。

如果 Pilot 没有产生可靠、可解释的正向行为，不进入20k正式训练。

## 12.1 正式训练条件

候选方案经过 Pilot 后，才开展三个种子的正式训练。

采用相同：

- Stage1 VFCE。
- TRAIN videos。
- Candidate generation。
- Legal assignment。
- Identity memory。
- Batch。
- Optimizer-update budget。
- 原生状态执行。

Seed：

`20261008`

`20261009`

`20261010`

保留最差 Seed20261009。

不能选择性删除种子。

## 12.2 多轮自主迭代

本任务允许多轮自主修复。

但每轮必须形成：

`Hypothesis → Code Change → Unit Test → Native Causal Test → Pilot → Formal Validation → Attribution`

只有当前一轮的失败原因被可靠定位，下一轮才能修改相应模块。

禁止仅靠增加 epochs、层数或随机搜索超参数反复重试。

如果第一轮 Commitment 改进失败：

根据真实错误分类决定继续修复 Candidate Reliability、历史记忆构建或 Identity Alias。

如第二轮仍失败：

重新检查监督可识别性、动作空间、基础视觉表示和 native lifecycle 设计。

如多轮对照显示同证据普通网络完全等价或更好：

如实报告方法没有独立 Jev 架构优势，不得强行宣称成功。

---

# 13. P10 — 完整在线原生验证

首先保证：

`GMT OFF parity`

`GTA-free throw test`

`Native ID assignment legality`

`State snapshot/restore`

`ID recycling`

`Gallery / Bank consistency`

`Cross-camera shared Global ID`

`Cache enabled/disabled equivalence`

`Deterministic run replay`

全部通过。

新增：

`CommitmentState serialization`

`CommitmentState identity integrity`

`Commitment continuation`

`Necessary correction`

`Unknown history abstention`

`Qualified fragment consolidation`

`Cross-camera continuation`

`No GT/future leakage`

不允许训练时使用一个理想 tracker，而推理时使用另一个真实 tracker。

模型训练和真实推理必须保持动作语义一致。

---

# 14. P11 — 开发集完整视频评测

使用现有冻结开发集：

video17、18、19。

并继续保留封存的：

video20、21、22。

不得自动开启 Full24 或官方 TEST。

真正执行：

`Current Images → Stage1 VFCE → Jev Identity Decision → Native Commit → Next Frame`

不能使用未来信息。

所有关键模型须输出：

- HOTA。
- AssA。
- IDF1。
- IDSW。
- MOTA。
- Frag。
- 官方 CVIDF1。
- 官方 CVMA。
- Correct continuation。
- Pure fragment switch。
- Polluted-history switch。
- Corrective switch。
- False merge。
- False split。
- False birth。
- Normal identity retention。
- UNKNOWN selected。
- Gallery contamination。

对比：

- Original GMT（独立完整系统）。
- Phase XIII Fixed。
- Phase XIV Fixed。
- Phase XIV Multi。
- Phase XIV Set Transformer。
- New Commitment-only。
- Fixed + same Commitment。
- Set Transformer + same Commitment。
- New Full Jev-MCMOT。

必须明确同证据对照和非同前端历史强基线的差别。

不能通过短轨过滤或 GT 重编号减少指标后冒充原始严格在线效果。

---

# 15. P12 — 成功门槛

建议将成功分成三个层次，而非只有一个含糊的 PASS。

## GO_TRACKING：JEV 确实完成可靠的在线 MCMOT

至少要求：

1. 完整原生 GTA-free 运行。
2. 不依赖 GT/未来帧。
3. 多摄像头身份约束合法。
4. 在三个种子上保持基本稳定。
5. HOTA 和 AssA 达到或超过同证据强关联控制器。
6. IDSW 显著低于 Phase XIV Multi 的391.667。
7. 不靠增加错误合并来人为减少 IDSW。
8. 没有通过大量错误出生把 IDSW 转换成身份碎片。
9. 跨相机 CVIDF1 不发生不可接受的明显退化。

最终挑战目标：

- HOTA 至少达到 Phase XIII Fixed 的75.637水平。
- 争取 IDSW 达到原冻结102.5门槛或更低。
- 争取保持或超过 Phase XIV Multi 已获得的88.546 CVIDF1。

这些数值是挑战目标，不可将不同历史实验直接当成同条件显著性证据。

正式科学 Gate 的具体同监督比较与容差必须事前冻结。

## GO_JEV_INDEPENDENT_VALUE

在 GO_TRACKING 之外，还必须证明：

- Full JEV 比同数据同预算的 Fixed Question有额外收益。
- 比同监督普通 Set Transformer有稳定正向收益。
- Commitment Question 不仅仅等价于固定滞后惩罚。
- Dynamic Question 与 OptionReader 在真实动作选择中产生正向贡献。
- 三种子之间没有严重不可解释的 IDSW 爆炸。
- 同监督消融支持具体架构结论。

没有该 Gate，就只能称为成功的 GTA-free learned MCMOT controller，不能宣称独立的 Jev 架构优势。

## GO_DEPLOYMENT

另外独立报告：

- Stage2 p95 ≤10ms。
- 完整双摄像头场景 FPS 目标≥25。
- Peak VRAM。
- Params/FLOPs。
- CPU/I/O/Detector/Policy 分项耗时。

Phase XIV 已显示完整系统前端也存在明显吞吐瓶颈。

因此算法的在线因果正确性与25FPS部署达标必须分别报告。

不得用忽略检测器的 Cache FPS 替代完整端到端 FPS。

不得修改历史25FPS门槛来追认成功。

---

# 16. P13 — 独立泛化问题

Phase XIV 已证实：

Stage1 已接触全部24个 VisionTrack TRAIN 视频。

因此新模型在这些视频上的结果不构成完全未见场景的端到端泛化证明。

本轮可继续使用 video17/18/19 做冻结协议下的开发评测。

但不得将反复开发的改进解释为独立测试提升。

如需真正独立验证：

优先核验外部数据与通用视觉前端预训练来源。

外部 WILDTRACK 当前结果较差，不能仅凭一个固定320样本前缀就证明严格跨域性能。

必须分开：

- Development performance。
- External scene transfer。
- Truly unseen full-system generalization。

若 clean Stage1 所需的模型初始化、训练数据、时间预算和实验资格能够合法满足，可以单独启动 clean frontend retraining。

若不满足，不允许重命名已有权重来冒充 clean training。

封存heldout及官方TEST仍需遵守研究数据资格；本次自主研究授权不代表允许绕过现有封存协议。

---

# 17. 必须新增的核心代码

建议新增：

`gtr/modeling/jev_phase15/commitment_state.py`

`gtr/modeling/jev_phase15/commitment_question.py`

`gtr/modeling/jev_phase15/commitment_option_reader.py`

`gtr/modeling/jev_phase15/candidate_reliability.py`

`gtr/modeling/jev_phase15/joint_action_ranker.py`

`gtr/modeling/jev_phase15/identity_fragment_controller.py`

`gtr/modeling/jev_phase15/native_commit_adapter.py`

`gtr/modeling/jev_phase15/model.py`

以及：

`reproduction_tools/jev_phase15_switch_forensics.py`

`reproduction_tools/jev_phase15_causal_commitment.py`

`reproduction_tools/jev_phase15_commitment_labels.py`

`reproduction_tools/jev_phase15_train.py`

`reproduction_tools/jev_phase15_onpolicy.py`

`reproduction_tools/jev_phase15_evaluate.py`

`reproduction_tools/jev_phase15_matlab.py`

`reproduction_tools/jev_phase15_latency.py`

`reproduction_tools/jev_phase15_finalize.py`

优先复用 Phase XIV 已经过验收的原生执行模块，不要无必要重写所有 tracker。

---

# 18. 最终科研报告

至少交付：

`docs/JEV_PHASE15_RESEARCH_GOAL.md`

`docs/JEV_PHASE15_PERSISTENT_IDENTITY_ARCHITECTURE.md`

`docs/JEV_PHASE15_CAUSAL_IDENTITY_COMMITMENT.md`

`docs/JEV_PHASE15_UNKNOWN_RELIABILITY.md`

`docs/JEV_PHASE15_TRAINING_AND_ABLATIONS.md`

`docs/JEV_PHASE15_FINAL_RESEARCH_REPORT.md`

报告目录：

`reports/JEV_PHASE15/`

至少包含：

- `FINAL_GOAL.json`
- `PHASE14_FROZEN_EVIDENCE.json`
- `SWITCH_EVENT_LEDGER.json`
- `SEED20261009_FAILURE_ATLAS.json`
- `COMMITMENT_NATIVE_CAUSAL_BRANCHES.json`
- `COMMITMENT_FEASIBILITY_GO_NO_GO.json`
- `COMMITMENT_LABEL_AUDIT.json`
- `UNKNOWN_RELIABILITY_AUDIT.json`
- `NATIVE_STATE_CONTRACT.json`
- `STRUCTURAL_TESTS.json`
- `TINY_RESULTS.json`
- `PILOT_RESULTS.json`
- `FORMAL_TRAINING_RESULTS.json`
- `FAIR_BASELINE_RESULTS.json`
- `ON_POLICY_STABILITY.json`
- `ONLINE_VALIDATION.json`
- `OFFICIAL_MATLAB_RESULTS.json`
- `EFFICIENCY.json`
- `GENERALIZATION_QUALIFICATION.json`
- `FINAL_GO_NO_GO.json`

每个文件必须绑定真正执行时的：

- Source SHA。
- Checkpoint SHA。
- Dataset SHA。
- Config SHA。
- Random Seed。
- Environment。
- Real evaluator。
- Input provenance。
- Scope。
- Result status。

未运行指标保持 null。

失败记录不得删除。

模型权重和大型轨迹留在服务器。

GitHub 仅推送必要源码、紧凑结果、报告与哈希清单。

---

# 19. Codex 自主执行策略

本次允许自主完成多阶段研究，不能每完成一个模块就以“是否继续”终止整个 Goal。

但必须遵守科研完整性：

**第一轮：复现和归因。**

明确哪些 IDSW 能通过合法 Commitment 避免，哪些不能。

**第二轮：实现和有界训练。**

建立 Q4 Commitment 与 Candidate Reliability，进行 Tiny/Pilot。

**第三轮：真实在线验证。**

测试完整视频，包含最差种子20261009。

**第四轮：针对剩余主要错误修正。**

根据真实数据决定是优化 Commitment、Candidate Reliability、Global Identity Memory，还是身份碎片管理。

**第五轮：公平对照与研究结论。**

检验真正来自 Jev 决策架构还是普通连续性正则化。

允许多个版本化的有界迭代，但每轮都要有独立可复现的假设和结果。不得无限制盲目训练或无声扩大计算预算。

普通软件错误自行修复。

科学数据资格不满足则停止不可信阶段。

如果相关模块被实验证明没有帮助，应该删除或保留为失败消融，而不是继续堆叠。

如果普通模型效果相同甚至更好，必须承认 Jev 独立价值尚未成立。

---

# 20. FINAL EXECUTION COMMAND

**现在立即执行 Phase XV。**

从：

`LYQ1107/WWW@0dc9607192c22bb4487b77b41de20f188caf7f1a`

创建独立 worktree。

执行：

`Frozen Phase XIV Audit`

→ `824 IDSW Failure Reconstruction`

→ `Pure Fragment / UNKNOWN Attribution`

→ `Legal Native Counterfactual Commitment`

→ `Persistent Identity State Design`

→ `Dynamic Q4 Commitment Question`

→ `Candidate Reliability`

→ `Commitment-aware Joint Assignment`

→ `Certified Continuation / Correction Labels`

→ `Tiny Learnability`

→ `Controlled Pilot`

→ `Formal Training`

→ `Balanced On-policy Training`

→ `Full Native MCMOT Validation`

→ `Worst-seed Regression`

→ `Ordinary Attention and Fixed Question Controls`

→ `Official Cross-camera Evaluation`

→ `Actual Runtime and Generalization Audit`

→ `Final GO/NO-GO`

不要只交付一个可以运行的新 Head。

不要只追求离线 WHO Accuracy。

不要把减少 Wrong-anchor 当成最终成功。

不要靠 GT、未来信息或重写历史 ID 来降低 IDSW。

不要把 `ABSTAIN`、`DEFER`、`START_NEW`、`REACTIVATE`、`CONTINUE` 混为一谈。

不要忽略已经发生的错误身份污染。

不要把同一 GT 的多个纯净预测 ID 无条件视为等价的最终提交动作。

**本次真正需要学习的是：识别这个人，同时知道该稳定延续哪个具体 Global ID，以及什么时候确实必须改变这个 ID。**

最后只有在真实严格在线 MCMOT 的身份稳定性、总体关联质量和公平架构对照均提供相应证据时，才允许宣布成功。

若达到 GO，提交完整可复现代码、Checkpoint Manifest、正式评测、消融与论文贡献总结。

若未达到 GO，必须给出具体仍然失败的身份事件、反事实证据及无法继续的科学阻塞，而不是笼统说“模型仍需提高”。

完成后推送 GitHub，返回最终 HEAD commit、全部科学 Gate 状态、真实 HOTA/AssA/IDF1/IDSW/CVIDF1/CVMA、速度、最差种子结果和下一步建议。

**FINAL SCIENTIFIC GOAL**

Build a genuinely reliable GTA-free Visual JEV for strictly online MCMOT, where identity recognition, persistent identity commitment, candidate reliability, and lawful global assignment jointly support stable multi-camera identity tracking.

**The goal is not to maximize short-term identity classification accuracy. It is to preserve correct identities over time without preventing necessary corrections.**
