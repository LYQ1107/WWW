# FINAL GOAL — WWW / JEV PHASE XIV
## Reliable Identity Decisions with Dynamic Questions and Constrained Online Learning

### 在 Phase XIII GTA-free JEV 的基础上，解决身份分裂、未知候选、Question 无效和真实泛化问题

**Repository:** https://github.com/LYQ1107/WWW

**Frozen starting commit:**

`2881fbdbca6501591470f9814d491a7aa8c1fb17`

**New branch:**

`jev/www-jev-phase14-reliable-identity-decisions-20261010`

## 0. 最终科研目标

本次继续坚持严格在线、多摄像头、GTA-free、真正 Jev 风格的结构化身份决策建模。

不要退回到 GMT GTA 后处理。

不要简单地训练一个可学习阈值。

不要继续无目标地增加 Transformer 层数、模型宽度或训练轮数。

本阶段必须解决以下三个科研问题：

**Goal A — Reliable Identity Assignment**

在保持现有 Global Identity Matching 能力的同时，减少错误关联、错误出生和身份碎片。

重点解决上一阶段：

- Full HOTA 75.056。
- Fixed Question HOTA 75.637。
- Full IDSW 118.667。
- 原冻结 IDSW gate 102.5。
- 自身状态训练后 Full HOTA 75.635，但 IDSW 143.667。
- TRAIN Wrong-anchor 与额外出生出现明显权衡。

**Goal B — Genuine Jev-style Question Value**

研究为什么动态 QuestionReader 未优于固定 Question。

重新设计具有真实、独立监督意义的多问题决策，而不是仅依靠一个 MATCH 类型的静态标签。

需要证明 Question 结构影响的是任务相关的决策证据，而不是增加参数或特征扰动。

**Goal C — Verifiable Generalization and Efficiency**

修复独立评估不足的问题，区分 Stage1 预训练暴露、关联网络开发验证、真正独立场景泛化。

独立优化 Stage2 的实时开销，并准确归因端到端瓶颈。

不得声称在原来已经接触全部24个训练视频的 Stage1 前端下，video20/21/22 构成完全独立的全系统测试。

---

# 1. P0 — 锁定 Phase XIII 的全部事实与科研资产

首先 checkout 指定提交，建立独立 worktree。

保护以下文件：

`docs/JEV_PHASE13_FINAL_RESEARCH_REPORT.md`

`reports/JEV_PHASE13/FINAL_GO_NO_GO.json`

`reports/JEV_PHASE13/TRAINING_PROTOCOL.json`

`reports/JEV_PHASE13/DENSE_DATASET_MANIFEST.json`

`reports/JEV_PHASE13/CANDIDATE_RECALL.json`

`reports/JEV_PHASE13/LABEL_AUDIT.json`

`reports/JEV_PHASE13/ARCHITECTURE_ABLATION.json`

`reports/JEV_PHASE13/ON_POLICY_TRAINING.json`

`reports/JEV_PHASE13/ONPOLICY_COMPARISON.json`

`reports/JEV_PHASE13/OFFICIAL_MATLAB_PRIMARY.json`

`reports/JEV_PHASE13/EFFICIENCY.json`

同时审计所有历史模型的 checkpoint SHA。

原始 Phase XIII 结果永久保留，不允许重新写入已有报告。

必须建立：

`reports/JEV_PHASE14/PHASE13_FROZEN_EVIDENCE.json`

其中明确记录：

- Original GMT 独立完整系统 HOTA78.110。
- Cosine HOTA69.085 / IDSW82。
- Full HOTA75.056 / IDSW118.667。
- Fixed Question HOTA75.637 / IDSW105。
- Set Transformer HOTA74.836。
- Full on-policy24k HOTA75.635 / IDSW143.667。
- 20k/24k 训练预算不同。
- Native GTA-free PASS。
- Full lifecycle supervision NOT_RUN。
- Heldout20/21/22 SEALED。
- Stage1 pretraining exposed all24 TRAIN videos。
- Historical G5/G7/G8/G9 scientific gates failed。

这份证据必须被后续所有新报告引用，不得覆盖。

---

# 2. P1 — 先做完整身份错误归因，不要立即重训

优先研究为什么 JEV 已具备较强匹配能力，但 IDSW 和身份出生依然不稳定。

必须利用 Phase XIII 已保存的真实 native predictions、committed IDs、Gallery、GT offline audit、actor journal 和 checkpoint。

不要先重新运行全部视频。

## 2.1 建立 Identity Error Taxonomy

将真实在线错误分解为：

A. WRONG_EXISTING_MATCH

错误关联到另一个人的已有 ID。

B. FALSE_SPLIT / FALSE_DEFER

真实已有身份可用，但模型拒绝并最终分裂成新 ID。

C. FALSE_MERGE

两个不同 GT 身份进入同一个 Global ID。

D. WRONG_REACTIVATION

错误恢复了 stale identity。

E. FALSE_BIRTH

真实身份已经在系统历史中出现过，却产生新的预测 ID。

F. GALLERY_CONTAMINATION

错误关联导致历史 Identity Memory 混合多个真实身份。

G. IDENTITY_UNAVAILABLE

正确身份不在合法候选集合中。

H. UNKNOWN_SUPERVISION

候选身份无法可靠认证，不能简单判作错误。

I. CROSS_CAMERA_ID_MISMATCH

同一真实身份跨摄像头被分配不同 Global IDs，或不同身份被误合并。

所有事件必须明确可验证的 GT/身份锚点来源。

不能把错误片段、IDSW、false birth 当成完全相同的统计量。

## 2.2 分析 IDSW 的真正来源

对 Full、Fixed Question、Set Transformer、Cosine、Full24k 进行逐事件对照。

必须回答：

- Full 的额外 IDSW 主要来自 false split 还是 false merge？
- 是否集中出现在 video16 类似的大候选场景？
- 哪些身份在连续多个 camera payload 中发生反复匹配？
- 是否存在同一个候选被反复 DEFER、然后重新关联？
- 错误第一次出现时，模型是否已经处于污染状态？
- 哪些问题出现在正确身份本来就缺失的时刻？
- 错误是否由跨相机历史引起，还是跨相机证据成功阻止了错误？
- 对比 Fixed Question，Full 的动态问题在哪些状态下改变最终分配？

输出错误类别数量、身份级持续区间、连续决策序列与典型可复现前缀。

## 2.3 UNKNOWN Candidate Forensics

现有训练含大量 UNKNOWN options。

必须核实：

`jev_phase13_learning.py::losses`

`jev_stage2/assignment.py::structured_assignment_loss`

`jev_stage2/native.py::commit`

之间的训练/部署候选语义。

定量回答：

- UNKNOWN candidate 进入最终匹配的比例。
- UNKNOWN 与已认证正确身份之间的 logit margin。
- UNKNOWN candidate 分数是否系统性偏高。
- Certified correct candidate 已存在时，UNKNOWN 是否经常抢占。
- UNKNOWN 是否比 NEW/DEFER 更容易获得错误优势。
- UNKNOWN 选择之后有多少出现真正可确认的污染。
- 候选数量从 10 增大至 50 时风险是否变化。

必须分别报告：

- Certified Correct。
- Certified Wrong。
- Selected UNKNOWN。
- Label Unavailable。
- Correct Candidate Missing。
- False Split。
- False Merge。

不准把 UNKNOWN 强行归入负例。

## P1 Gate

在上述统计完成之前，不启动新的20k正式训练。

输出：

`docs/JEV_PHASE14_IDENTITY_ERROR_TAXONOMY.md`

`reports/JEV_PHASE14/ERROR_ATTRIBUTION.json`

`reports/JEV_PHASE14/UNKNOWN_CHOICE_FORENSICS.json`

`reports/JEV_PHASE14/FALSE_MERGE_SPLIT_DIAGNOSTICS.json`

---

# 3. P2 — 研究 Dynamic Question 为什么无效

这是本次 JEV 架构科研的核心。

首先审计：

`gtr/modeling/jev_stage2/model.py`

具体函数：

`GlobalIdentityJev.encode_state`

`GlobalIdentityJev.encode_questions`

`GlobalIdentityJev.score_questions`

特别注意：

当前正式监督主要是 MATCH。

因此即使代码具有三个 task embeddings，也不意味着真正进行了三任务 Jev 学习。

## 3.1 分离 Question 的不同作用

必须测试：

1. Static Question Type。
2. Dynamic Detection-conditioned Question。
3. Dynamic Identity-availability Question。
4. Dynamic Candidate-conflict Question。
5. Dynamic Question with cross-camera identity evidence。
6. Multiple typed questions using shared state。

每种结构应该有真实的信息区别。

禁止简单将多个固定向量重复六次后称为多问题推理。

## 3.2 建立三个可以验证的研究问题

### Q1 — WHO

当前检测最可能对应哪个合法 Global ID？

监督来自完整的自然身份关联数据。

### Q2 — IDENTITY AVAILABLE?

当前真正正确的历史身份是否存在于此时合法的候选集合？

对于确切可认证的 positive candidate，可以明确给出 AVAILABLE=1。

对于可靠证实候选集合不包含正确身份的状态，可以给出 AVAILABLE=0。

历史身份混合、GT 不明确、候选身份无法对应时，标记 UNKNOWN。

不得将没有发现正确身份直接当成身份不存在。

### Q3 — EVIDENCE TRUST / IDENTITY CONTAMINATION

某个历史身份表示是否存在可认证的纯净/污染证据？

从 TRAIN predicted histories 和 GT offline attribution 构造可靠监督。

只对具有明确历史一致性证据的样本给予标签。

缺失或模糊的身份历史继续保持 UNKNOWN。

这三个问题可以共享视觉状态和 OptionReader，但监督、输出概率和评估范围必须分别定义。

**如果 Q2/Q3 不能提供足够真实且非平凡的监督，则必须记录资格失败，不得仅靠 Q1 重命名制造三问题结构收益。**

## 3.3 必须建立同证据、同容量普通模型对照

至少包括：

- Fixed Question JEV。
- Full Phase XIII JEV。
- Multi-question JEV。
- Set Transformer + 同样的多任务标签。
- MOTIP-style + 同样的多任务标签。
- 普通共享 MLP heads + 同样的多任务标签。

必须保证全部方法拥有相同输入证据、标签范围和训练预算。

真正值得证明的是：

**多问题的 Jev 风格结构化决策，在真实身份可用性和候选选择方面，比普通共享网络产生稳定的独立收益。**

而不仅是 Question embeddings 有梯度。

---

# 4. P3 — 在不伪造身份的条件下研究身份缺席监督

目前真实 NEW 和 REACT 正例过少。

可以设计训练侧的合法候选集合干预实验，但要严格区分真实数据和人工问题。

## 4.1 Certified Candidate Withholding

从 TRAIN 中选取已可靠证明当前正确身份存在于候选集合的事件。

保持：

- 当前检测真实视觉特征不变。
- 历史所有其他候选不变。
- 当前真实在线前缀不变。
- 当前相机和时间不变。

仅在一个版本化的训练输入副本中，从合法 Option 集合移除已认证正确候选。

这样构造一个：

`Correct identity absent from provided options`

的受控问题。

它的合格动作是：

`DEFER`

而不是自动：

`START_NEW`

这种样本用于训练 Question 识别当前答案集合是否包含正确身份。

不能将其计为自然发生的 NEW 或 Stale Recovery。

保留未干预的正常样本作为 paired control。

## 4.2 难度分级

事前冻结：

- Easy absence。
- Hard absence。
- Many-candidate absence。
- Cross-camera absence。
- Ambiguous presence。
- Correct candidate present。

为每个分组报告：

- Natural/Intervened 来源。
- 标签可认证性。
- Positive/Negative support。
- False absence。
- False presence。
- DEFER accuracy。
- Incorrect match rate。
- False split rate。

不允许仅使用容易区分的缺席样本训练后报告大幅提升。

## 4.3 学习目标

可以考虑：

`L = L_identity_choice + λ_avail L_identity_availability + λ_struct L_assignment`

其中 availability 是一个真正的类型化概率问题。

关键验证：

当模型判断正确身份不在当前集合时，是否减少错误合并。

同时，正确身份存在时，不应频繁拒绝造成身份分裂。

这是对当前两个相反错误的直接控制。

训练数据只使用 TRAIN，绝不使用开发视频的 GT 构建新的训练干预。

---

# 5. P4 — 从普通身份 CE 转向可靠的联合动作训练

此前训练：

`choice CE + structured assignment + Brier`

仍未充分保证 false merge/split 的平衡。

这一阶段不建议直接修改推理阈值来掩盖问题。

应该从真实、可确认的身份动作后果出发，考虑代价敏感的结构化训练。

## 5.1 明确动作代价

至少分开：

- Wrong existing assignment。
- False merge。
- False split。
- False birth。
- Wrong stale recovery。
- Correct existing continuation。

动作代价系数必须在 TRAIN 审计与先验研究目标上冻结。

不能看开发视频 HOTA 后随意修改，以获得有利数字。

## 5.2 Joint Assignment Training

使用完整合法联合匹配。

不能单纯把每条边当作独立分类问题。

可以比较：

A. Standard Choice CE。

B. Existing structured assignment loss。

C. Cost-sensitive structured assignment loss。

D. Typed availability + joint assignment。

E. Availability + structure + limited on-policy。

同一损失应提供给 JEV 和普通 Set Transformer，以区分架构贡献与监督目标贡献。

## 5.3 On-policy Improvement

先诊断 Phase XIII 的 4k 自身状态训练为什么：

- Wrong-anchor 减少。
- HOTA 小幅上升。
- False births 增加。
- IDSW 增加。

新的 On-policy 改进不得只增加训练 updates。

应明确训练更多以下状态：

- 正常正确身份连续状态。
- 真实 false split 前缀。
- 真实 false merge 前缀。
- 污染历史下仍有可靠候选的状态。
- Correct identity absent 的状态。
- 可可靠恢复 identity 的状态。

使用按身份组与时序段划分的训练采样，避免大量重复相邻帧主导梯度。

必须保留相同 update budget 的旧训练目标和新训练目标对照。

不能因为新模型多训练4k次，就把其结果直接归因于新损失。

---

# 6. P5 — 重新审查长期视觉记忆和跨摄像头证据

Phase XIII 说明长程与跨相机信息组合很重要。

但现有消融不够细。

必须拆分：

- Recent appearance。
- Global historical mean。
- Own-camera mean。
- Other-camera mean。
- Camera metadata。
- Time-since-last-observed。
- Geometry/motion。
- Global candidate counts。
- Shared State Encoder。

进行逐项与部分因子组合消融。

特别是：

`no_cross_camera`

当前同时去除了多类跨相机/全局信息。

`no_long_term`

当前保留 recent、去除多个历史摘要。

需要新的细粒度实验判断，究竟是视觉历史、相机特定原型、时序信息，还是全局身份统计起主要作用。

初期优先使用冻结 checkpoint 的 inference-time controlled ablation 和少量配对前缀，之后只有具有明确价值的条件才进行重新训练。

凡是训练时没见过的特征删除，只能解释为依赖性诊断；不能用它替代经过重训的正式结构消融。

输出：

`reports/JEV_PHASE14/IDENTITY_MEMORY_FACTORIAL.json`

`reports/JEV_PHASE14/CROSS_CAMERA_EVIDENCE_AUDIT.json`

---

# 7. P6 — 重新定义实时性实验的责任边界

保护 Phase XIII 的历史效率 Gate：

- Stage2 p95 ≤10ms。
- 双摄像头场景 ≥25FPS。
- Phase XIII G8 FAIL。

不能修改旧 Gate 或追认其通过。

但必须认识到，Cosine 基线在同一前端下也只有约4.603场景FPS。

因此新研究应分别审计：

A. Stage1 image loading。

B. Backbone/Detector。

C. VFCE ROI extraction。

D. History token construction。

E. QuestionReader。

F. OptionReader。

G. Global assignment。

H. Native ID/Memory commit。

I. CUDA synchronization / Python overhead。

## 7.1 优先做无数值变化优化

重点检查：

`gtr/modeling/jev_stage2/memory.py`

`gtr/modeling/jev_stage2/native.py`

`gtr/modeling/jev_stage2/model.py`

目前每次构造候选会重新遍历身份历史并计算 Gallery mean 等统计量。

研究是否能安全维护增量的：

- Running visual sum/count。
- Own-camera prototype。
- Other-camera prototype。
- Cached identity history tokens。
- Batched metadata。
- Reused projected historical features。

不得因缓存而破坏 native commit 语义。

必须进行：

- 完整状态 parity。
- Cache invalidation。
- Resume parity。
- ID recycling parity。
- Cross-camera memory parity。
- Numeric difference。
- Full online ID parity。

如果浮点求和顺序导致不能 bitwise 相等，应明确数值误差和决策稳定性证明；不能在没有核验的情况下宣称完全等价。

## 7.2 分开评测速度

每个模型都报告：

- Stage2 policy-only p50/p95。
- History build p50/p95。
- Total Stage2 p50/p95。
- Full image-to-tracks FPS。
- Camera-payload FPS。
- Scene-frame FPS。
- Peak VRAM。

验证真正去掉 dormant GTA 权重后是否能够释放显存，不得把仅删除参数带来的变化夸大为实时 FPS 提升。

优先在同硬件、同图像、同前端和同线程设置下重复测试。

如果 25FPS 完整系统目标仍然不可达，应继续记录 FAIL，同时指出前端的可量化瓶颈。

不能通过排除检测器时间“实现25FPS”。

---

# 8. P7 — 解决 Stage1 预训练暴露导致的独立验证问题

这一任务非常重要。

Phase XIII 的 Stage1 checkpoint 曾在全部24个 VisionTrack TRAIN 视频上训练。

因此现有开发与封存视频对 Stage1 而言都不是完全未见过的场景。

必须先完成以下审计：

- Stage1 训练视频列表。
- Stage2 训练视频列表。
- Development 视频。
- Heldout 视频。
- 实际视觉前端训练暴露。
- Scene/identity overlap。
- 外部数据是否可用。
- 是否存在真正未参与训练的带 GT MCMOT 视频。

规划两个独立验证方案。

## Scheme A — 新外部数据

优先检查服务器已有数据。

如需新增，优先使用可直接获取的公开数据、官方镜像或百度网盘直连，不优先依赖代理。

冻结外部数据身份与评估协议。

不得查看新评估场景结果后改超参数。

如果目标只做 GTA-free 模型泛化，不允许混入官方 TEST 进行选择。

## Scheme B — Clean Stage1 Retraining

如果资源足够且确有必要，从可核验的通用视觉初始化重新训练 Stage1。

保证新训练集不包含预定的开发和独立评估场景。

随后为原始 GMT、普通关联基线和 JEV 重新生成相同前端特征、完成公平 Stage2 训练。

不能用已经接触评估场景的 Stage1 权重冒充 clean retrain。

这一任务计算成本较高，必须先有独立实验预算、可用数据和训练配置 Gate。

只有验证条件真正满足，才可提出未见场景的泛化结论。

原 video20/21/22 继续 SEALED。

Full24/Official TEST 未获得新授权前不得启动。

---

# 9. P8 — 新 Phase XIV 的实验矩阵

不要直接重新运行12种架构 ×3 seeds ×20k updates。

首先复用所有合法的 Phase XIII 冻结模型开展无训练诊断。

然后执行小规模受控实验，再将真正有证据的方案扩展到正式训练。

建议核心方法为：

B0：Frozen Phase XIII Full JEV20k。

B1：Frozen Phase XIII Fixed Question20k。

B2：Frozen Set Transformer20k。

B3：Full JEV + Identity Availability Question。

B4：Full JEV + Certified Candidate Withholding。

B5：Full JEV + Merge/Split Cost-sensitive Assignment。

B6：Full JEV + Availability + Structured Assignment。

B7：B6 + Balanced On-policy Training。

B8：同监督普通 Set Transformer。

B9：同监督 MOTIP-style 或匹配容量普通网络。

每次训练必须有对应控制条件。

实验划分：

## Diagnostic

不训练，使用冻结权重和真实前缀确定问题性质。

## Pilot

每个候选方案使用事前固定的有界训练预算，验证正确性、数值稳定与动作行为。

## Formal

只有通过 Pilot 的少数方案进行完整三种子训练。

主训练预算必须与对应基线一致。

On-policy 额外 updates 必须有相同额外预算的普通训练控制组。

不要以 BEST checkpoint 在测试视频上的 HOTA 选择模型。

---

# 10. P9 — 科学 Gate

建立独立的 Phase XIV `PREREGISTRATION.json`。

冻结所有评价标准后才能查看新实验结果。

## G0 — Native Truth

保留历史 OFF/GTA-free、完整原生状态、RNG、Gallery 与 ID recycling 测试。

## G1 — Error Taxonomy

所有高频错误归因明确，UNKNOWN 不被错误认证。

## G2 — Dynamic Question Learnability

新问题不仅有梯度，而且具有独立真实监督与不同合法动作空间。

没有 Q2/Q3 有效标签则不得称为真实多问题 Jev 训练。

## G3 — Availability and Action Semantics

DEFER、NEW、REACT、ABSTAIN 严格区分。

训练侧候选删除不能被报告为自然 NEW/REACT 正例。

## G4 — Stability

必须同时观察：

- HOTA。
- AssA。
- IDF1。
- IDSW。
- False merge。
- False split。
- False birth。
- Normal association retention。

不允许仅凭 Wrong-anchor 减少宣布成功。

历史 G5 的 IDSW 102.5 门槛保持作为固定可比科学门槛，不得降低后追认旧结果。

## G5 — Question Independent Value

必须与 Fixed Question 和普通 Set Transformer 使用相同证据、任务标签、参数预算和求解器。

Full JEV 的独立收益需要同时体现于可靠决策质量和在线关联指标。

任何仅靠多训练或多输入获得的收益须单独归因。

## G6 — Cross-camera Identity

分别报告官方 MATLAB CVIDF1、CVMA、跨视角身份一致性与错误恢复。

不得用单摄像头 HOTA 替代跨相机评价。

## G7 — Efficiency

Stage2 延迟与端到端 FPS 分开判定。

历史25FPS目标不能被静默删除。

新的算法相对速度比较必须同前端、同硬件和同样本。

## G8 — Independent Evidence

预训练暴露限制未解决时，不允许主张独立全系统泛化。

Heldout20/21/22 保持封存。

Official TEST 和 Full24 禁止自动开启。

---

# 11. P10 — 最终交付

新增目录：

`docs/JEV_PHASE14_*`

`reports/JEV_PHASE14/`

`reproduction_tools/jev_phase14_*`

建议至少完成：

`docs/JEV_PHASE14_FINAL_GOAL.md`

`docs/JEV_PHASE14_ERROR_ANALYSIS.md`

`docs/JEV_PHASE14_QUESTION_REDESIGN.md`

`docs/JEV_PHASE14_DENSE_ACTION_SUPERVISION.md`

`docs/JEV_PHASE14_GENERALIZATION_PROTOCOL.md`

`docs/JEV_PHASE14_FINAL_RESEARCH_REPORT.md`

和：

`reports/JEV_PHASE14/PREREGISTRATION.json`

`reports/JEV_PHASE14/PHASE13_FROZEN_EVIDENCE.json`

`reports/JEV_PHASE14/ERROR_ATTRIBUTION.json`

`reports/JEV_PHASE14/UNKNOWN_CHOICE_FORENSICS.json`

`reports/JEV_PHASE14/FALSE_MERGE_SPLIT_DIAGNOSTICS.json`

`reports/JEV_PHASE14/QUESTION_SUPERVISION_ELIGIBILITY.json`

`reports/JEV_PHASE14/AVAILABILITY_INTERVENTION_DATA.json`

`reports/JEV_PHASE14/STRUCTURED_ACTION_LOSS_ABLATION.json`

`reports/JEV_PHASE14/IDENTITY_MEMORY_FACTORIAL.json`

`reports/JEV_PHASE14/ON_POLICY_STABILITY.json`

`reports/JEV_PHASE14/FAIR_BASELINE_RESULTS.json`

`reports/JEV_PHASE14/ONLINE_VALIDATION.json`

`reports/JEV_PHASE14/OFFICIAL_MATLAB_RESULTS.json`

`reports/JEV_PHASE14/LATENCY_ATTRIBUTION.json`

`reports/JEV_PHASE14/PRETRAIN_EXPOSURE_AUDIT.json`

`reports/JEV_PHASE14/FINAL_GO_NO_GO.json`

所有报告必须记录真实 source/data/checkpoint/config SHA、训练种子、合法输入范围、实验状态与失败证据。

不能伪造缺失指标。

大文件和 checkpoint 保留服务器。

GitHub 只提交必要代码、文档、紧凑结果与图表。

---

# 12. 最终必须回答的问题

1. Phase XIII 的额外 IDSW 主要来自 false split 还是 false merge？
2. UNKNOWN candidate 在真实关联中究竟造成什么影响？
3. 为什么 Full 动态 Question 不如 Fixed Question？
4. 动态 Question 在真实多个问题下是否具有独立价值？
5. 是否能学习可靠的 Identity Availability，而不把 UNKNOWN 当负例？
6. 是否能减少 false split，同时不增加 false merge？
7. 当前 On-policy Training 为什么减少 Wrong-anchor 却增加额外出生？
8. 跨相机历史视觉和全局统计各自有多大贡献？
9. 长期视觉信息具体哪些部分不可替代？
10. 真正的 Jev-style 模型是否优于同监督的普通 Set Transformer？
11. GTA-free 与原生身份管理是否仍然满足全部在线因果契约？
12. 当前系统的 Stage2 与完整端到端耗时瓶颈分别是什么？
13. 是否已具备真正独立的未见场景验证条件？
14. 是否值得继续走完整 MATCH–REACT–MEMORY 三问题共享学习？
15. 根据本轮真实结果，最可信的 WWW 论文贡献究竟是什么？

回答必须基于实际完成的实验。

不能因为研究目标是 JEV，就预设 JEV 一定优于普通 Attention。

---

# FINAL EXECUTION COMMAND

现在从：

`LYQ1107/WWW@2881fbdbca6501591470f9814d491a7aa8c1fb17`

建立独立 Phase XIV worktree 与研究分支。

按以下顺序执行：

**Frozen Evidence Audit**

→ **Identity Error Taxonomy**

→ **UNKNOWN Candidate Forensics**

→ **Dynamic Question Diagnostics**

→ **Identity Availability Supervision**

→ **Certified Candidate Interventions**

→ **Merge/Split Cost-sensitive Learning**

→ **Balanced On-policy Training**

→ **Fine-grained Identity Memory Ablation**

→ **Fair Set Transformer / Fixed Question Comparison**

→ **Strict Online Development Validation**

→ **MATLAB Cross-camera Evaluation**

→ **Stage2/Frontend Latency Attribution**

→ **Independent Generalization Qualification**

→ **Final GO/NO-GO**

普通工程问题自行修复后继续。

不要完成诊断就自动停止；只要数据与科学 Gate 合法，应继续完成有界 Pilot 和必要的正式训练。

但不允许未经验证直接启动大量无差别训练，也不允许为了得到更高 HOTA 改动历史 Gate、偷用 heldout 或制造生命周期标签。

本阶段重点不再是证明 JEV 可以替换 GTA——Phase XIII 已经完成了这项工程目标。

**本阶段最终目标是证明或证伪：真正具有多问题身份可用性建模、动态候选决策和可靠动作风险训练的 Jev-style MCMOT，能否在严格在线条件下改善长期身份稳定性，并产生普通 Attention 无法仅靠相同监督解释的额外价值。**

完成后推送 GitHub，提供准确的 HEAD SHA、完成的实验矩阵、失败原因、所有 GO/NO-GO 结果与下一阶段建议。
