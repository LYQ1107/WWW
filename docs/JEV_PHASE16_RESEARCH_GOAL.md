# FINAL GOAL — WWW / JEV PHASE XVI
## Evidence Recovery, Safe Memory Admission and Reliable Identity Decisions
### 面向严格在线 MCMOT 的历史身份恢复、安全记忆与可验证 JEV 决策学习

**Repository:** https://github.com/LYQ1107/WWW

**Frozen starting commit:**

`3e16cbf914d70bbde0d999faabc85afc9b5ae43f`

**新研究分支：**

`jev/www-jev-phase16-evidence-recovery-20261011`

**最终科研目标：**

在 Phase XV 已经完成的 GTA-free Visual JEV、Persistent Commitment 和原生 Global Identity Management 基础上，解决以下核心问题：

> 当正确身份在当前候选集合中无法可靠确认，或者原有身份历史已经被错误观测污染时，系统能否利用严格过去的视觉证据，安全恢复正确的全局身份，并避免错误进一步传播？

本轮不再把提高 WHO Accuracy、减少 IDSW 或增加 Commitment 权重作为单独的成功条件。

**必须同时优化：正确身份恢复、身份连续性、错误纠正、历史记忆安全性和完整视频 MCMOT 质量。**

---

# 一、必须冻结的历史事实

首先审计：

`docs/JEV_PHASE15_FINAL_RESEARCH_REPORT.md`

`reports/JEV_PHASE15/FINAL_GO_NO_GO.json`

`reports/JEV_PHASE15/PILOT_RESULTS.json`

`reports/JEV_PHASE15/ALL_QUERY_RELIABILITY_V3.json`

`reports/JEV_PHASE15/NATIVE_UNSAFE_EVENTS.json`

`reports/JEV_PHASE15/IDENTITY_ERROR_PROPAGATION.json`

`reports/JEV_PHASE15/COMMITMENT_LABEL_AUDIT.json`

`reports/JEV_PHASE15/COMMITMENT_FEASIBILITY_GO_NO_GO.json`

### Phase XV 的真实研究结论

- GTA-free native execution：PASS。
- Persistent Commitment mechanism：存在局部正向因果证据。
- 三轮 Tiny/Pilot：全部完成。
- 正式三种子20k：NOT_RUN。
- Matched4k：NOT_RUN。
- GO_TRACKING：FAIL。
- GO_JEV_INDEPENDENT_VALUE：NOT_ESTABLISHED。
- GO_DEPLOYMENT：FAIL。

第三版 V3、seed20261009、完整 DEV17/18/19：

- HOTA：74.188。
- AssA：74.891。
- IDF1：90.521。
- IDSW：147。
- CVIDF1：90.286。
- CVMA：86.781。
- MOTA：89.659。

历史最差 seed IDSW 从824降到147，但不能单凭它认为身份跟踪成功。

真实 Pilot 表明：

- V1 必要纠错监督不足。
- V2 必要纠错5/5，但存在有害原生未来分支。
- V3 必要纠错5/7，仍存在有害分支。
- 全查询认证不足。
- 没有满足正式训练安全资格。

禁止覆盖这些结果。

---

# 二、本轮研究优先级与边界

按照以下优先级执行：

**P0：正确身份候选与历史证据可用性审计。**

**P1：历史视觉证据恢复。**

**P2：安全的视觉记忆写入与污染隔离。**

**P3：真实原生未来的安全收益验证。**

**P4：重新设计 JEV 的身份兼容性与恢复决策。**

**P5：训练、原生评测与公平对照。**

不要将这六项同时实现后再测试。

每一项必须根据上一项的真实结果决定是否继续。

本轮优先保持 Stage1 VFCE、检测器、Global ID 编号、原生 Gallery/Bank 合法状态以及基础 WHO 架构不变。

不要重新训练检测器，不要增加无必要的 Transformer 层，也不要立即进行三种子20k正式训练。

---

# 三、P0 — 先找出正确身份究竟丢失在哪里

这是整个 Phase XVI 最重要的前置研究。

目前的 `IdentityMemory.history_values()` 将一个 ID 的历史主要表示为：

1. 最后一次视觉观测。
2. 全历史平均视觉特征。
3. 当前相机平均视觉特征。
4. 另一个相机平均视觉特征。

当历史身份混入其他人的视觉特征时，这种压缩可能掩盖局部仍然可靠的历史观测。

但必须先验证，不能预设所有候选缺失都是均值压缩导致的。

## P0.1 实际候选缺失分类

针对 Phase XV 的所有真实高风险查询，将候选缺失分为：

**A — ABSENT_FROM_ALL_HISTORY**

在当前时刻以前，所有已有历史记录中都不存在可认证的正确身份观测。

**B — PRESENT_IN_RAW_GALLERY**

正确身份历史观测存在于真实原始 Gallery，但当前摘要特征无法有效表达。

**C — PRESENT_IN_STALE_BANK**

正确身份已进入历史 Bank，但当前原生恢复流程没有将它提供给 MATCH/REACT 候选集合。

**D — CANDIDATE_ELIGIBILITY_FAILURE**

正确 ID 存在，但因为合法候选构造、时序状态或生命周期规则没有进入实际可选集合。

**E — CANDIDATE_PRESENT_BUT_WRONG_SELECTION**

正确候选已经合法可选，但 JEV 选择了另一个身份。

**F — HISTORY_CONTAMINATED**

候选包含正确人物的历史观测，但也包含其他人物，无法作为全局纯净身份认证。

**G — AMBIGUOUS**

GT、观测对应或历史锚点不足以判定。

特别注意：

“没有纯净可认证候选”不等于“正确历史身份不存在”。

不得将 F/G 强制标记为错误候选。

## P0.2 分析任务

使用原始 TRAIN 和历史开发诊断前缀，统计：

- 正确身份实际出现过的比例。
- Raw Gallery 中存在正确历史证据的比例。
- 历史摘要表示丢失可区分证据的比例。
- Stale Bank 候选遗漏比例。
- Candidate eligibility 错误比例。
- 已有可认证候选的错误选择比例。
- 完全不可辨识状态的比例。
- 这些状态与错误合并、出生和身份切换的关系。

重点审计：

`gtr/modeling/jev_stage2/memory.py`

`gtr/modeling/jev_stage2/native.py`

`gtr/modeling/jev_phase15/commitment_state.py`

`gtr/modeling/jev_phase15/candidate_reliability.py`

以及原始 GMT 的 Bank promotion 和 recovery 逻辑。

必须核实 `react_learned=False` 时使用的真实恢复路径、候选范围、分数计算以及最终 START_NEW 语义。

## P0.3 硬性研究判断

如果大量正确身份观测存在于 Raw Gallery，却无法通过四个历史摘要被有效读取：

优先研究多原型历史表示。

如果主要问题是 Stale Bank 和候选资格：

优先修复合法候选检索与生命周期范围。

如果真实历史中根本没有相关观测：

承认当前信息不足，不要求 JEV 凭空恢复不存在的证据。

如果正确候选本来就存在：

优先研究选择决策，不增加候选集合复杂度。

输出：

`EVIDENCE_AVAILABILITY_AUDIT.json`

`RAW_GALLERY_RECOVERABILITY.json`

`CANDIDATE_FAILURE_TAXONOMY.json`

`PHASE16_P0_GO_NO_GO.json`

**P0 未完成，不允许启动新的正式训练。**

---

# 四、P1 — 建立多原型、可追溯的历史身份视觉表示

只有 P0 证明历史摘要丢失有效证据时，才执行本模块。

提出：

**Causal Multi-Prototype Identity Evidence Memory**

每个 Global ID 不再只依赖一个长期平均视觉向量，而是能够保存有限数量的、由过去真实观测构成的局部视觉证据。

## P1.1 设计

每个 Global ID 的历史证据可以划分为：

- Recent observations。
- Historical appearance prototypes。
- Camera-specific prototypes。
- Temporally coherent segments。
- Provisional observations。
- Uncertain observations。

这些历史片段必须从真实过去的检测与已提交历史生成。

不能根据 GT 进行在线纯净片段筛选。

不能使用未来观测重新划分过去已经输出的身份。

建议从现有1024D Stage1 VFCE开始，不训练新视觉 backbone。

## P1.2 多原型构造

可以调查：

1. 时间连续的局部片段聚合。
2. 同摄像头近期视觉一致性聚合。
3. 轻量视觉聚类。
4. 基于外观变化的在线 Segment Boundary。
5. 有界长度的代表性历史观测保存。

所有方法必须：

- 具有严格因果性。
- 对历史 Token 数设定上限。
- 支持跨摄像头身份记忆。
- 支持空历史。
- 支持被污染历史。
- 支持长时间失踪。
- 支持 Global ID 复用与恢复。
- 支持 snapshot/restore。
- 不改变已输出身份编号。

### 核心约束

**历史 Segment 不是新的 Global ID。**

多个 Segment 可以属于同一个已提交的 Global ID。

JEV 可以读取哪个 Segment 支持当前匹配，但最终必须提交合法的 Global ID，而不是随意创建 Segment ID。

否则只是把原来的 ID fragment 问题搬进新网络。

## P1.3 先做冻结模型实验

保持 Phase XV WHO/JEV 权重冻结。

比较：

A. 原始四个视觉摘要。

B. 多个近期观测。

C. 时序 Segment prototypes。

D. Camera-aware segment prototypes。

E. 上述证据的有界组合。

必须评价：

- 正确身份候选支持率。
- Existing ID compatibility。
- GT 可确认时的 positive retrieval recall。
- False candidate activation。
- 污染历史中的候选可靠性。
- 跨摄像头对应。
- 实际前向延迟。
- Gallery 内存占用。

如果增加更多历史 Token 只增加计算，却没有改善可认证证据覆盖，则停止该方案。

不得依靠增加 Top-K 后静默丢弃困难候选。

输出：

`MULTI_PROTOTYPE_IDENTITY_MEMORY.json`

`HISTORY_SEGMENT_RETRIEVAL.json`

`CANDIDATE_RECOVERY_ABLATION.json`

---

# 五、P2 — 设计 Safe Memory Admission，阻止错误证据不断强化

现有原生执行中，已提交身份会更新 Gallery 与历史统计。

Phase XV 已经证明，某些正确的第一步身份选择之后，后续仍可能产生错误写入与污染传播。

因此这一模块必须独立验证。

## P2.1 区分身份提交和视觉证据认证

身份提交：

`COMMIT_ID`

是当前真实在线动作。

视觉证据录入：

`ADMIT_EVIDENCE`

决定当前视觉观测是否立即成为未来长期身份匹配的可靠证据。

它们不能简单视为同一个决策。

一个检测被赋予 ID17，不代表这次视觉观测已经有足够证据用于永久修改 ID17 的长期视觉原型。

## P2.2 建立三层记忆

建议：

**CONFIRMED**

有足够的过去视觉一致性与时序证据支持的历史观测。

**PROVISIONAL**

已经真实提交，但尚未具备充分证据的新观测。

**QUARANTINED**

当前与历史显著冲突或来源不确定，不立即参与可靠原型更新的观测。

这里的 CONFIRMED 是算法的运行时证据状态，不是 GT 纯净认证。

不能宣称 CONFIRMED 一定对应正确人物。

## P2.3 必须维持原生身份完整性

不允许：

- 删除实际历史输出。
- 用未来 GT 修复过去的 ID。
- 静默修改已经提交的 Global ID。
- 因为隔离视觉特征而让 Gallery/hits 不一致。
- 删除已经发生的错误记录。
- 用 GT 决定哪条观测进入 CONFIRMED。

可以维护独立的匹配证据视图，让长期稳定原型不被尚未确认的观测立即覆盖，但保留完整、可审计的原始提交日志。

明确区分：

`Raw Committed Gallery`

与：

`Trusted Matching Evidence View`

不能借“记忆隔离”之名重写既有 native state。

## P2.4 首先采用不训练的受控策略

暂不训练 MEMORY JEV Head。

因为当前没有合格的真实 WRITE/KEEP 长期效用监督。

先测试严格过去证据规则产生的安全记忆策略，并与：

- Always Write。
- Recent-only prototype。
- Conservative provisional admission。
- Delayed confirmation。
- Bounded quarantine。

进行公平比较。

之后只有在真实合法动作后果支持时，才考虑 JEV 学习证据写入决策。

输出：

`SAFE_MEMORY_ADMISSION_PROTOCOL.json`

`EVIDENCE_QUARANTINE_TESTS.json`

`NATIVE_MEMORY_STATE_PARITY.json`

---

# 六、P3 — 实施真正能定位问题的配对因果实验

本轮必须区分以下两种可能：

**错误来源于当前身份选错。**

或者：

**当前身份可能选对，但错误写入改变了后续历史。**

这两种问题不能再用同一套 H32 改善数字混合评价。

## P3.1 固定模型，隔离因素

对于同一个真实 TRAIN prefix 和冻结 JEV：

实验 A：

`Original Decision + Original Memory`

实验 B：

`Original Decision + Safe Evidence Admission`

实验 C：

`Evidence-aware Decision + Original Memory`

实验 D：

`Evidence-aware Decision + Safe Evidence Admission`

除指定干预外，保持：

- 完全相同的过去原生状态。
- 相同真实检测。
- 相同 Stage1 VFCE。
- 相同 RNG。
- 相同后续策略。
- 相同合法候选规则。

如果分支中动作状态变化导致未来输入发生变化，必须保留这种真实反馈，不能人为同步后续身份历史。

## P3.2 未来窗口

先比较：

H8、H16、H32。

需要验证长尾错误时再执行有限 H64。

主指标：

- Future CLEAR IDSW。
- New cross-GT identity mixing。
- Wrong-owner observations。
- False merge。
- False split。
- False birth。
- Cross-camera inconsistency。
- Writes into polluted matching evidence。
- Correct identity recovery。
- Recovery latency in observed frames。

所有统计必须说明独立事件数、共享前缀和聚类方式。

相邻帧重叠窗口不能当作独立样本重复计数。

## P3.3 重点反例

必须包含：

`TRAIN12 frame3 camera1 row6`

以及 Phase XV 所有有害原生窗口。

不能只测试有利窗口。

重点验证：

**安全证据隔离是否能够减少后续身份混合，而不仅是减少短期 IDSW。**

如果只减少 IDSW，却增加错误混合或错误持续时间，必须判定相关实验没有通过安全收益门槛。

输出：

`MATCH_VS_MEMORY_CAUSAL_FACTORIAL.json`

`RECOVERY_AND_CONTAMINATION_EFFECTS.json`

`NATIVE_FUTURE_SAFETY_GO_NO_GO.json`

---

# 七、P4 — 再设计 JEV 的 Identity Recovery 决策

只有 P0–P3 产生真实可用的证据后，才开始修改 JEV 网络。

保留：

- WHO。
- Availability。
- Trust。
- Persistent Commitment。

新模型核心增加：

**Evidence-conditioned Identity Recovery**

它不是单纯给历史 ID 增加一个新分类头。

而是让 JEV 读取真正的局部历史视觉证据、候选可用性和当前身份承诺状态。

## P4.1 具体问题

Q1：

`WHO IS THIS OBSERVATION COMPATIBLE WITH?`

Q2：

`IS A RELIABLE EXISTING IDENTITY AVAILABLE?`

Q3：

`WHICH HISTORICAL EVIDENCE SUPPORTS THIS IDENTITY?`

Q4：

`WHICH PERSISTENT GLOBAL ID SHOULD BE COMMITTED?`

Q5：

`IS RECOVERY OR DEFER NECESSARY BEFORE CREATING A NEW ID?`

这些问题必须存在真实、可区分的监督或诊断价值。

不能只增加五个 Task Embeddings 就宣称完成五问题学习。

Q5 在真实监督不足时，只能使用冻结规则或明确 UNTRAINED 状态，不允许未经训练的 logits 直接覆盖 native recovery。

## P4.2 Evidence-conditioned OptionReader

对每个实际身份候选，动态读取：

- Current visual tokens。
- Recent prototype。
- Historical local segment tokens。
- Cross-camera evidence。
- CommitmentState。
- Candidate uncertainty。
- Candidate competition。

输出身份选项的兼容性与风险。

禁止依赖 GTA association logits。

必须在新的 JEV DIRECT 模式下重新通过 GTA throw-mock 测试。

## P4.3 明确合法动作

至少区分：

`MATCH_EXISTING_ID`

`DEFER_TO_RECOVERY`

`REACTIVATE_EXISTING_STALE_ID`

`START_NEW`

`UNRESOLVED_EVIDENCE`

其中 `UNRESOLVED_EVIDENCE` 如果没有实际可执行的 native 状态语义，不能直接作为模型输出动作。

必须先说明未解决证据最终如何映射到当前帧的合法输出、暂存与身份状态，不允许通过丢弃检测或延迟未来帧变相提高指标。

尤其注意：现有 MATCH DEFER 后续可能通过 Bank 或 START_NEW 完成处理，不能把这种 DEFER 直接视为真正安全的拒绝决策。

---

# 八、P5 — 训练数据必须先通过资格审计

当前自然 REACT 正例只有9条，MEMORY WRITE/KEEP 真实长期效用标签不足。

不得直接训练声称完整生命周期的 JEV。

先从 TRAIN12/13/14/16 实际在线状态中构建：

`Identity Evidence Recovery Dataset`

包含：

1. Pure existing support。
2. Polluted existing support。
3. Raw Gallery clean-segment support。
4. Stale history support。
5. Correct ID missing。
6. False existing association。
7. Safe continuation。
8. Necessary correction。
9. New identity。
10. Ambiguous/UNKNOWN。

GT 只允许用于训练期离线标签审计。

在线推理不得读取任何 GT 信息。

对于同一个 GT 的多个预测历史碎片：

WHO 可以保留 multiple positives。

具体 Commitment 应以当前可认证的连续性、纠错需求和动作后果进行监督。

不能简单让所有同 GT 的历史 ID 获得相同 Commitment target。

## P5.1 样本资格门槛

正式训练前检查：

- 是否有真正安全恢复正例。
- 是否有不安全恢复负例。
- 是否有污染历史且存在局部可靠视觉证据的事件。
- 是否有需要纠正当前错误 ID 的事件。
- 是否有正确的正常持续关联。
- 是否有足够独立 identity/time clusters。
- 是否有独立 reserved TRAIN temporal blocks。

如果只有大量安全延续正例，而纠错正例极少：

不能因为总体 accuracy 很高就进入正式训练。

如果某种动作没有真实监督：

不得制造该动作标签。

---

# 九、P6 — 有界训练与自主迭代

建议采用三个有明确假设的研究版本。

## V1 — Evidence Recovery Only

冻结 Phase XV JEV 权重。

仅修改历史证据读取和候选恢复。

目的：

验证历史原始证据是否能够弥补被污染均值造成的身份候选缺失。

不得增加新网络容量。

## V2 — Recovery + Safe Memory

冻结同一模型。

新增可审计的安全证据写入机制。

目的：

验证减少污染写入能否改善真实后续身份状态。

必须与 V1 进行同前缀配对。

## V3 — Learned Evidence-conditioned JEV

只有 V1/V2 存在安全正向因果证据、且训练监督合格时，才训练新 JEV。

以 Phase XV 现有权重为初始化候选。

重新训练：

- 动态 Evidence Reader。
- Candidate Reliability。
- Identity Commitment。
- 必要时的合法恢复决策。

保持普通 Fixed Question/Set Transformer 拥有相同历史证据和监督。

## 训练预算

- 无训练因果验证优先。
- Tiny：128–256 updates。
- Pilot：1000–1500 updates。
- 先以 seed20261009 检查历史最差情形。
- 无安全 Pilot PASS，不启动长训。
- Formal：合格后20k updates、三个种子。
- Matched controls：同数据、同预算。
- On-policy 4k：仅在基础 Formal 与真实状态资格通过后执行。

不允许反复随机重训直到出现一个好种子。

优先在训练视频的真实 native histories 上做有界迭代。

三版仍失败时，生成证据驱动的 Scientific Blocker Report，不无声扩大预算。

---

# 十、P7 — 公平对照必须回答真正的科研问题

正式训练至少需要：

A. Phase XV Frozen JEV。

B. Frozen JEV + Multi-prototype evidence。

C. Frozen JEV + Safe Memory。

D. Multi-prototype + Safe Memory。

E. Ordinary Set Transformer + same evidence and supervision。

F. Fixed Question + same evidence and supervision。

G. New Evidence-conditioned JEV。

如果新方法效果提高，需要明确区分：

- 是更多历史视觉信息带来的收益？
- 是记忆隔离带来的收益？
- 是新的训练监督带来的收益？
- 还是动态 Jev Question/Option 结构本身带来的收益？

不得把普通结构也能获得的改善包装成 Jev 独立价值。

---

# 十一、P8 — 原生闭环与完整视频评价

所有候选必须执行：

`Actual Image → Stage1 VFCE → Evidence Retrieval → JEV Decision → Native Commit → Updated Memory → Next Frame`

禁止：

- GT teacher forcing。
- Future-frame information。
- Hidden GTA calls。
- GT-based identity aliasing。
- Historical ID renumbering。
- 使用开发集 GT 决定在线动作。
- 用缓存推理 FPS 冒充真实图像 FPS。

完整报告：

- HOTA。
- AssA。
- IDF1。
- MOTA。
- IDSW。
- Frag。
- 官方 CVIDF1。
- 官方 CVMA。
- False merge。
- False split。
- False birth。
- Necessary correction recall。
- Wrong-owner observation duration。
- Gallery contamination。
- Safe recovery precision/recall。
- UNKNOWN selection risk。
- Normal association retention。

必须保留每个种子、每个视频的结果。

严格使用 pooled TrackEval 和未修改的官方 MATLAB 评测。

不能只展示最优 seed。

---

# 十二、P9 — 科研成功条件

设置独立的三个目标。

## GO_TRACKING

必须：

- Native GTA-free PASS。
- Identity state restore PASS。
- 不使用未来或GT输入。
- 三种子完整训练与评价合格。
- HOTA/AssA 不低于最强同证据普通控制器。
- IDSW 不再出现 Phase XIV 那样的824次极端失稳。
- 相比同监督控制器，错误身份混合、错误出生不增加。
- 不通过延长错误身份持续时间换取更低 IDSW。
- 跨相机身份结果不明显退化。

保持 Phase XV 冻结参考：

- 每种子 IDSW≤250。
- 三种子均值 IDSW≤195.8335。

HOTA75.637、IDSW102.5和 CVIDF1 88.546 作为进一步挑战目标，而非允许修改历史结果后追认的标准。

## GO_JEV_INDEPENDENT_VALUE

在 GO_TRACKING 基础上：

- 与 Fixed Question 同监督比较。
- 与 Set Transformer 同监督比较。
- 动态 Question/OptionReader 消融有实际正向效果。
- 正向收益不仅来自历史证据或 Memory 规则。
- 三种子没有反向严重失稳。
- 报告预注册的配对指标与效应量。

未满足时可报告 GTA-free MCMOT 工程成功，但不能宣称 Jev 架构创新通过。

## GO_DEPLOYMENT

继续保留历史：

- Stage2 p95≤10ms。
- 完整双摄像头 scene FPS≥25。

当前 Phase XV Stage2 p95约69ms，完整场景FPS约2.6–2.7。

因此必须重点测量：

- 新历史原型检索成本。
- Memory update。
- JEV forward。
- Hungarian。
- Bank recovery。
- 原始图像读取。
- Backbone/Detector。
- VFCE。

不要新增大量历史 Tokens 后忽略实际延迟。

如果完整系统瓶颈主要来自 Stage1，必须如实说明。

跟踪科学质量与部署门槛分别判定。

---

# 十三、P10 — 严格保护数据边界

沿用：

TRAIN：12/13/14/16。

Development：17/18/19。

SEALED：20/21/22。

Full24：NOT_AUTHORIZED。

Official TEST：NOT_AUTHORIZED。

现有 Stage1 已接触 VisionTrack全部24个 TRAIN 视频，因此开发验证只能作为已有前端下的关联控制器研究，不得声称完全未见的全系统泛化。

开发集可以在事前冻结的协议中执行确认，但不能根据结果反复修改训练标签或动作门槛。

要证明独立泛化，应另行完成真正无暴露的数据与前端资格。

---

# 十四、建议新增代码

重点新增：

`gtr/modeling/jev_phase16/evidence_ledger.py`

`gtr/modeling/jev_phase16/history_segment_encoder.py`

`gtr/modeling/jev_phase16/multi_prototype_memory.py`

`gtr/modeling/jev_phase16/candidate_retrieval.py`

`gtr/modeling/jev_phase16/safe_memory_admission.py`

`gtr/modeling/jev_phase16/evidence_conditioned_reader.py`

`gtr/modeling/jev_phase16/identity_recovery_policy.py`

`gtr/modeling/jev_phase16/native_adapter.py`

以及：

`reproduction_tools/jev_phase16_candidate_audit.py`

`reproduction_tools/jev_phase16_recoverability.py`

`reproduction_tools/jev_phase16_memory_causal_test.py`

`reproduction_tools/jev_phase16_training_data.py`

`reproduction_tools/jev_phase16_train.py`

`reproduction_tools/jev_phase16_evaluate.py`

`reproduction_tools/jev_phase16_matlab.py`

`reproduction_tools/jev_phase16_latency.py`

`reproduction_tools/jev_phase16_finalize.py`

必须先检查是否能够复用已存在的 Phase XIII–XV 代码，避免无必要重写。

---

# 十五、交付要求

新增：

`docs/JEV_PHASE16_RESEARCH_GOAL.md`

`docs/JEV_PHASE16_EVIDENCE_RECOVERY.md`

`docs/JEV_PHASE16_SAFE_MEMORY_DESIGN.md`

`docs/JEV_PHASE16_NATIVE_CAUSAL_PROTOCOL.md`

`docs/JEV_PHASE16_TRAINING_PROTOCOL.md`

`docs/JEV_PHASE16_FINAL_RESEARCH_REPORT.md`

以及：

`reports/JEV_PHASE16/FINAL_GOAL.json`

`reports/JEV_PHASE16/PREREGISTRATION.json`

`reports/JEV_PHASE16/EVIDENCE_AVAILABILITY_AUDIT.json`

`reports/JEV_PHASE16/RAW_GALLERY_RECOVERABILITY.json`

`reports/JEV_PHASE16/CANDIDATE_FAILURE_TAXONOMY.json`

`reports/JEV_PHASE16/MULTI_PROTOTYPE_RESULTS.json`

`reports/JEV_PHASE16/SAFE_MEMORY_ADMISSION.json`

`reports/JEV_PHASE16/NATIVE_MEMORY_PARITY.json`

`reports/JEV_PHASE16/MATCH_VS_MEMORY_CAUSAL_FACTORIAL.json`

`reports/JEV_PHASE16/LABEL_ELIGIBILITY.json`

`reports/JEV_PHASE16/TINY_RESULTS.json`

`reports/JEV_PHASE16/PILOT_RESULTS.json`

`reports/JEV_PHASE16/FORMAL_TRAINING_RESULTS.json`

`reports/JEV_PHASE16/FAIR_BASELINE_RESULTS.json`

`reports/JEV_PHASE16/ONLINE_VALIDATION.json`

`reports/JEV_PHASE16/OFFICIAL_MATLAB_RESULTS.json`

`reports/JEV_PHASE16/LATENCY_RESULTS.json`

`reports/JEV_PHASE16/FINAL_GO_NO_GO.json`

所有实验必须绑定实际运行时：

- Git SHA。
- Dataset SHA。
- Checkpoint SHA。
- Config SHA。
- Seed。
- Native state SHA。
- Evaluator。
- Runtime environment。
- Scope。

未运行指标保持 null。

不得将 `NOT_RUN` 填成 0。

权重、视频、长轨迹和完整日志留服务器，GitHub只推送代码、配置、紧凑指标、研究报告和 SHA 清单。

---

# 十六、Codex 自主执行与停止规则

本次不要求你每完成一个有界研究阶段就询问是否继续。

只要仍在本轮冻结研究范围内，且具备可信的科学实验条件，可以自主推进。

但必须满足：

1. 每个新版本有独立明确假设。
2. 新版本修改点与上一轮失败原因直接对应。
3. 优先使用无训练实验验证机制。
4. 训练标签必须真实可认证。
5. 不修改既有冻结 Gate。
6. 不挑选有利种子。
7. 不隐瞒有害分支。
8. 不突破预注册计算/存储预算。
9. 不自动打开 sealed heldout/official TEST。
10. 不把“工程完成”标记为“科研成功”。

允许本阶段三个主要研究版本。

若某版本通过原生因果安全 Pilot，则无需再次询问，直接按照预注册预算启动正式训练与公平比较。

若连续三版没有合格的身份恢复收益，或确认可观测历史本身缺乏足够信息，则停止不可信扩展，并明确：

- 哪类历史身份不可恢复。
- 为什么不可恢复。
- 需要什么新的真实视觉/时空证据。
- 是否需要修改 Stage1 ReID。
- 是否需要新数据集。
- 是否值得继续研究 JEV。
- 是否应转向更可靠的普通关联架构。

不要为了坚持 JEV 方向而掩盖结构负结果。

---

# FINAL EXECUTION INSTRUCTION

现在立即从：

`LYQ1107/WWW@3e16cbf914d70bbde0d999faabc85afc9b5ae43f`

建立新的 Phase XVI worktree。

先完整审计已有实验和真实代码。

然后按顺序执行：

**P0 — Candidate Availability and Failure Taxonomy**

→ **P1 — Raw Gallery Recoverability**

→ **P2 — Causal Multi-Prototype Identity Evidence**

→ **P3 — Safe Memory Admission**

→ **P4 — Native Factorial Counterfactual Tests**

→ **P5 — Evidence-conditioned JEV Recovery Design**

→ **P6 — Qualified Tiny/Pilot Training**

→ **P7 — Safe Native Closed-loop Evaluation**

→ **P8 — Three-seed Formal Training if Qualified**

→ **P9 — Fair Fixed/Set Transformer Comparison**

→ **P10 — Official MATLAB MCMOT Evaluation**

→ **P11 — Real Online Latency**

→ **P12 — Final Scientific GO/NO-GO**

**本次不允许通过增加训练步数掩盖历史身份证据缺失。**

**本次不允许通过强制延续旧 ID 掩盖错误身份合并。**

**本次不允许通过把 UNKNOWN 当负例制造虚假恢复能力。**

**本次必须先证明历史身份信息可以被合法恢复，再训练 JEV 利用这些信息。**

最终目标：

**A strictly online GTA-free Visual JEV that can recognize an identity, maintain a correct persistent Global ID, detect insufficient or contaminated evidence, and safely recover without propagating identity errors into future memory.**

研究结束后，提交并推送代码，返回最终 commit、每阶段状态、真实错误归因、完整在线指标、最差种子表现、正式实验是否获得资格，以及明确的科研结论。
