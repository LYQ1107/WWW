# Codex 执行任务书：JEVTrack 一周科研冲刺

## 0. 你的身份与最终目标

你现在是本项目的首席算法研究工程师、实验执行者和科研证据审核者。

项目地址：

https://github.com/LYQ1107/WWW

目标会议：WWW 2027

执行周期：2026 年 10 月 12 日至 10 月 18 日，优先保障论文主要实验和摘要所需证据。2026 年 10 月 25 日为全文提交截止。

项目暂定名称：

**JEVTrack: Reliable Identity Association Across Cameras**

中文：

**JEVTrack：跨摄像头稳定身份跟踪**

这是工作名称。若论文证据最终不能支持 JEV 的独立贡献，不得为了保留名称而夸大结果。

### FINAL GOAL

在现有 GMT 多摄像头多目标跟踪系统上，研究并实现一个能够有效利用跨摄像头历史身份信息的 JEV 关联模块。

最终证明或否定以下假设：

**在完全相同的视觉输入、身份历史、合法关联范围和在线状态转移条件下，JEV 能否比原始 GMT、固定规则、可学习阈值、普通可学习融合以及普通 Attention 更准确、更稳定地维护跨摄像头 Global ID，同时保持合理的模型规模与运行速度？**

你的目标不只是生成报告，而是完成真实模型修改、训练、公平对照、完整在线跟踪实验、效率测试和论文贡献验证。

所有研究结论以实测数据为准，不能事先假定 JEV 一定胜出。

---

## 1. 论文创新点：重新组织研究内涵

本论文必须始终围绕：

**Online Multi-Camera Multi-Object Tracking（在线多摄像头多目标跟踪）**

不得将研究逐渐转换成普通单摄像头 MOT、一般分类问题、静态 ReID 检索或通用多选题任务。

### 核心创新方向一：跨摄像头身份一致性

英文工作术语：

Cross-Camera Identity Consistency

我们关心的不是简单增加跨摄像头视觉特征，因为 GMT 已经具有 CFCE 和 GTA。

我们真正希望研究：

同一个 Global ID 在不同摄像头中是否得到了相互支持的历史证据；当不同摄像头的观测出现冲突时，模型是否能够更准确地识别正确身份。

具体能力：

- 利用当前摄像头及其他摄像头的真实历史观测。
- 区分外观相似性和历史身份一致性。
- 在两个相似人物产生竞争时利用跨摄像头证据。
- 不把错误历史的高相似度当作可靠身份归属。
- 在保持真实跨相机身份一致的同时，减少错误合并。

必须与 GMT 自身的跨视角建模对比，说明新模块到底额外解决了什么。

如果该能力仅由增加跨摄像头特征带来，而普通 MLP 或 Attention 同样有效，必须如实归因。

### 核心创新方向二：长期身份稳定性

英文工作术语：

Persistent Identity Consistency

研究目标：

减少 Global ID 的不必要切换、身份碎片化和错误历史传播。

必须区分：

- 当前匹配是否正确。
- 原始 Global ID 是否长期稳定。
- 历史是否混入过其他人物。
- 保持当前身份是否有足够证据。
- 调整一次关联是否可能破坏其他正确关联。

不要直接训练一个缺乏真实监督的长期未来价值模型。

本周首先使用已有身份 GT 和真实过去历史进行可监督学习，再通过原生在线回放验证长期效果。

不能将某个历史身份外观相似直接等价于长期身份正确。

### 第三个创新方向：有条件保留

可能的方向：

Efficient Shared Decision Inference

仅当模型能够证明共享状态下的多问题执行具有独立收益，而且参数量、真实计算时间与 FPS 可以接受时，才将它写为第三个创新点。

如果没有证据，不要硬凑第三个创新。

论文原则上保留两个有实证支持的贡献，比三个未经证明的贡献更好。

---

## 2. 必须永远回答的五个审稿人问题

建立：

docs/JEV_ONEWEEK/REVIEWER_FIVE_QUESTIONS.md

此文件作为所有实验设计的强制检查项。

### Q1：JEV 和二次关联有什么区别？

审稿人可能问：

“你们的效果是不是仅仅因为 GMT 第一次匹配之后又执行了一次关联？”

你必须设计真实同条件对照：

A. 原始 GMT。

B. GMT + 原始规则二次关联。

C. GMT + 再运行一次 Hungarian。

D. GMT + 学习式二次关联。

E. GMT + JEV。

另外增加必要的 2×2 机制实验：

原始评分 / JEV 评分 × 原始求解流程 / 相同额外求解流程。

如果新方法需要产生多个合法关联方案，则所有强对照也必须获得相同的方案和计算预算。

必须分离：

新增评分信息的作用；
额外求解次数的作用；
候选集合改变的作用；
历史状态改变的作用。

不能把多运行一次 Hungarian 的收益归因于 JEV。

### Q2：为什么不用固定阈值或可学习阈值？

复用 Phase VI 已实现的动态门限模型及相关实验。

必须比较：

- 固定关联阈值。
- 根据当前状态预测的动态阈值。
- 根据每一对检测和身份预测的阈值。
- Binary JEV。
- 新 JEV。

所有方法使用相同的历史证据与合法关联范围。

分析新 JEV 是否能够在两个分数均很高的合法 Global ID 之间作出更正确的区分，而不仅仅改变接受或拒绝关联的门槛。

如果新 JEV 在绝大多数事件上的结果可以由一个可学习阈值重现，则不得声称实现了独立的结构化身份推理能力。

### Q3：为什么不用几个可学习参数、MLP 或普通 Attention？

必须实现或复用：

- 固定融合权重。
- 可学习线性融合权重。
- 小型 MLP。
- DeepSets。
- Set Transformer / 标准 Attention。
- Phase XIII Fixed Question。
- Phase XIV Multi-question JEV。
- 新 JEV。

其中固定和可学习融合至少使用：

appearance；
geometry / motion；
历史连续性；
跨摄像头历史一致性。

MLP、DeepSets、Attention 和 JEV 必须尽量匹配：

训练样本；
监督标签；
真实视觉证据；
当前检测；
历史轨迹；
候选集合；
全局分配约束；
训练更新；
参数规模和计算预算。

不允许让 JEV 使用额外监督，却让 MLP 只学习普通匹配标签。

必须回答：

新方法的优势来自新的证据、新的监督，还是 JEV 的结构？

### Q4：引入 JEV 到底增加多少参数、多少延迟、损失多少 FPS？

要求精确测量。

已知历史参考：

Phase XIV Multi-question 注册参数：4,231,173。

Phase XV F_full 推算注册参数：5,923,209。

Phase XIV 已记录卸载的旧 GTA 相关参数：21,750,016。

以上均需重新实测与校验。

必须提供四套统计：

1. 原始完整 GMT 的总参数和活跃参数。
2. 现有 JEV 替换方案的总参数和活跃参数。
3. 新 JEV 的总参数和活跃参数。
4. 新增模块相对实际被替换模块的净参数变化。

同时提供：

trainable parameters；
registered parameters；
actual forward-active parameters；
MACs/FLOPs；
峰值显存；
单次模型推理时间；
历史证据构建时间；
求解器时间；
整个 Stage2 p50/p95；
完整真实双摄像头场景 FPS。

不要将原论文 A6000 上的 14.6 FPS 与当前 V100 上的 3–4 FPS 直接相减作为 JEV 开销。

必须在同一服务器、同一 GPU、同一批真实视频、相同线程设置和计时方式下比较。

新结构设计目标：

新增方案推理模块尽可能不超过 0.5M 参数；
整体 FPS 相对最强同条件基线损失目标不超过 10%。

这两个数是本周设计目标，不是已有实测结论。

若达不到，报告实际成本以及其对应的准确率收益，不得隐瞒。

### Q5：我们的模型究竟是不是真正的 JEV 思想？

不能因为网络中存在 Attention、Question Embedding、多个 Head，就直接宣称完整实现 JEV。

需要检查以下结构契约：

**State**
同一时刻的完整当前检测和跨摄像头身份历史，只编码一次。

**Question**
至少存在两个含义不同、能够用真实监督或可靠标注验证的问题。

建议：

问题1：当前身份在不同摄像头之间是否一致？

问题2：当前历史身份是否有足够证据支持继续沿用？

允许使用内部 Choice、Noul、Score 等类型化决策接口，但不要求通过自然语言运行。

**Options**
动态构造的合法身份或合法完整分配结果，不能只是固定数量的分类标签。

**Decision**
输出可验证的相对分数或概率，并保留不确定性。

**Shared Computation**
多个问题确实复用共享状态，而不是分别完整计算两遍。

必须做：

- Dynamic Question vs Fixed Question。
- 两个问题 vs 单问题。
- 去掉共享状态。
- 去掉问题条件化 Reader。
- 去掉跨摄像头证据。
- 去掉身份稳定性证据。
- 候选顺序置换。
- 候选数量变化。
- 同状态不同问题的输出敏感性。
- 置信度校准。
- 多问题计算开销对比。

如果模型只剩下单一匹配分数和固定 Attention，或者多个问题不能产生独立可验证的输出，应当在报告中标为 JEV-inspired Attention，而不是声称完整的 JEV 决策实现。

不得宣称复现 TypeSafe 未公开的商业内部架构。

---

## 3. 第一阶段：全面复用历史资产，不重复造轮子

先检查仓库全部历史 Phase VI–XXI 相关报告、代码、checkpoint manifest 和 native 状态。

重点路径：

docs/JEV_PHASE6_FINAL_REPORT.md
docs/PHASE7_FINAL_RESEARCH_REPORT.md
docs/PHASE10_FINAL_RESEARCH_REPORT.md
docs/JEV_PHASE12_FINAL_RESEARCH_REPORT.md
docs/JEV_PHASE13_FINAL_RESEARCH_REPORT.md
docs/JEV_PHASE14_FINAL_RESEARCH_REPORT.md
docs/JEV_PHASE15_FINAL_RESEARCH_REPORT.md
docs/PHASE17_FINAL_RESEARCH_REPORT.md
docs/PHASE19_FINAL_RESEARCH_REPORT.md
docs/PHASE20_FINAL_RESEARCH_REPORT.md
docs/WWW_JEV_MASTER_REQUIREMENTS_AUDIT.md

现有 MASTER 分支：

jev/www-jev-master-requirements-audit-20261011

已核对的参考提交：

Phase XIV：
d1c11faac27e3e67bf51a15448257c74eca44bbe

Phase XV：
3e16cbf914d70bbde0d999faabc85afc9b5ae43f

Phase XVI：
666e2b922eb676b3552db0c6dc8b67b1b3978450

最新已确认 MASTER 审计：
020a9b90bfce98141cf90f8e342230e5eeac23d2

执行前检查远端是否存在更新，报告实际起点 SHA。

不得覆盖这些冻结结果。

建立新分支：

jev/www-jev-oneweek-20261012

建立新的独立工作树，优先使用：

/home/liuyeqiang/WWW_jev_oneweek_20261012

若该目录存在，先核实是否属于本任务；不要强制覆盖。

建立复用登记表：

reports/JEV_ONEWEEK/REUSED_EVIDENCE_REGISTRY.json

每项实验标记：

REUSE_VALID：
协议、权重、输入与实验问题一致，可直接复用。

REUSE_DIAGNOSTIC：
可用于方法动机或分析，但不能作为正式主结果。

RERUN_REQUIRED：
需要同条件重新运行。

REJECTED：
历史实验失败，不应作为本周新方案。

NOT_AVAILABLE：
服务器缺少必要原始资产。

严禁以历史同名指标替代当前统一实验。

### 必须保留的历史事实

Phase VII 中固定规则曾优于 JEV。

Phase XIII 的 Fixed Question 三种子 HOTA75.637，高于 Full75.056。

Phase XIV Multi-question：
HOTA73.505；
AssA73.639；
CVIDF1 88.546；
IDSW391.667。

Phase XV V3：
HOTA74.188；
AssA74.891；
CVIDF1 90.286；
IDSW147。

Phase XVI hard memory：
HOTA61.764；
IDSW1494。

这些来自不同训练阶段或策略，不能简单组合成一张“完全公平的最佳成绩表”。

Phase XIV Multi 的 seed20261009 曾出现824次 IDSW，不得从历史记录中排除。

原始 GMT 的完整系统开发集参考 HOTA78.110，同期不同前端及关联流程的差异必须披露。

优先复用已存在的模型权重和缓存，不盲目重新训练之前的 20k 模型。

---

## 4. 第二阶段：确定最终模型基座

明确采用：

**Phase XIV 算法结构 + Phase XV 原生运行接口。**

具体文件：

gtr/modeling/jev_stage2/model.py
gtr/modeling/jev_stage2/assignment.py
gtr/modeling/jev_stage2/memory.py
gtr/modeling/jev_stage2/native.py

gtr/modeling/jev_phase14/model.py
gtr/modeling/jev_phase14/memory.py

gtr/modeling/jev_phase15/model.py
gtr/modeling/jev_phase15/commitment_state.py
gtr/modeling/jev_phase15/native_commit_adapter.py

算法主体从 Phase XIV 的 GlobalIdentityJev 与 ReliableIdentityPolicy 出发。

采用 Phase XV 的原生 Gallery、Bank、Instances、ID commit、状态快照、回放和评价机制。

Phase XV V3 整套 Q4 + CandidateReliability 只作为冻结参考及可选初始化模块，不默认完整保留。

禁止本周启用：

Phase XVI hard quarantine；
Phase XVII 未通过安全门的软记忆策略；
未经证明的学习式 MEMORY WRITE；
未经证明的统一 MATCH/REACT/MEMORY 生命周期模型。

必须保留：

原始 GMT Stage1；
相同 ReID 特征；
相同跨摄像头来源；
原生 Hungarian；
每摄像头身份容量约束；
真实 Gallery 和 Bank；
合法的旧身份恢复与新 ID 创建；
实际完整在线状态转移。

跨摄像头允许同一 Global ID 合法出现，不能错误地把每摄像头的身份唯一性约束扩张为跨摄像头同一时间只允许一个目标使用该 ID。

---

## 5. 第三阶段：本周的新 JEV 结构

不要简单复制此前的门控残差形式。

新模块内部可以使用合法联合分配的评价，但论文创新名称仍为：

跨摄像头身份一致性；
长期身份稳定性。

新模块建议由三部分组成。

### A. Shared Identity State

复用现有 1024D ReID 输入和 256D 共享状态表示。

显式分离：

Current Observation；
Own-Camera History；
Other-Camera History；
Global Identity History；
Temporal / Geometry Evidence。

利用已有四槽历史视觉表示，不额外依赖全帧大视觉语言模型。

所有历史必须真实来自当前时间之前已提交的观测。

### B. Cross-Camera Consistency Reader

针对当前检测和各个合法历史身份，读取跨摄像头一致性证据。

通过已有 TRAIN GT 构造监督：

同一真实身份的跨摄像头观测为正支持；

不同真实身份、外观容易混淆的历史为困难负样本；

身份归属混合、GT重复或无法认证的样本必须保留 UNKNOWN mask。

同一 Global ID 历史混入不同人物时，不能直接以其最近的高相似度观测作为正确归属证明。

本模块必须输出独立、可评价的跨摄像头一致性结果，而不只是一个不可解释的隐藏特征。

### C. Identity Stability Reader

负责学习当前 Global ID 历史是否具有可靠的连续支持。

输入：

历史连续性；
多视角一致性；
身份历史中的相互冲突；
实际可观察的前驱支持；
当前检测之间的身份竞争关系。

其训练目标是保护正确的身份延续，并在证据可靠时改善错误关联。

不能把身份纯净程度、外观相似度和身份动作长期安全性混成同一个真值标签。

如监督缺失，相关输出保持 UNKNOWN，不训练伪造风险标签。

### D. 对实际关联的作用

新模型可以评价当前原生完整分配和少量合法替代分配。

内部使用：

Shared State
→ Typed Questions
→ Cross-Camera Evidence
→ Identity Stability Evidence
→ Legal Assignment Evaluation
→ Native Commit

要与阈值区别开，新模块应能够比较**多个相互竞争的 Global ID 分配关系**，而不是只决定原匹配分数是否超过门槛。

但也不能为了制造复杂结构而暴力枚举全部排列。

建议：

- 基线原分配必须始终包含。
- 只针对真实存在身份竞争的场景生成替代。
- 最多四组分配供第一版研究。
- 替代方案必须由当前合法边和原生身份容量约束生成。
- 不使用 GT 或未来帧生成或筛选部署时的方案。
- 所有强对照获得完全相同的方案集合。
- 记录正确完整分配是否存在于可评方案集合。
- 没有可靠替代时保留原始分配。

评分应包含真实的联合关系或竞争上下文。

如果评分可以完全分解成独立边分数之和，然后只是重新调用 Hungarian，则不能将其宣传为具有独立非加性交互的联合决策网络。

必须比较加性边评分与存在实际候选交互的版本。

新模块初始化应当保留原策略输出；例如新评分残差初始化为零。

训练前验证 OFF 和初始新结构的关联结果完全一致。

禁止让随机初始化的新模块在第一轮就直接破坏已训练系统。

---

## 6. 第四阶段：直接使用已有数据训练

只有一周时间，原则上不下载新的大型数据集。

优先使用现有已获准的 TRAIN12/13/14/16、已缓存原生轨迹和人物身份 GT。

复用：

reproduction_tools/jev_phase15_collect_dataset.py
reproduction_tools/jev_phase15_collect_train.py
reproduction_tools/jev_phase15_losses.py
reproduction_tools/jev_phase15_train.py

以及 Phase XIV 已有的监督构造和训练脚本。

建立一个单独的 MATCH-only 研究协议。

重要范围限制：

原 MASTER 的8 TRAIN/3 RESERVED 是针对长期安全纠错控制器的独立安全监督门槛。

本次不能将普通 MATCH GT 训练伪装成满足长期安全动作监督。

本周新的训练协议仅授权利用原已许可的 TRAIN 范围开展受控关联学习和验证。

历史扩展到 video02–09、预留10/11/15 的授权原本只用于数据审计和有界因果检查，不自动扩张为任意训练使用。

VisionTrack20/21/22、官方 TEST、Full24 继续封存。

所有数据分割、模型选择、时间块、人物依赖关系和训练预算必须在本轮新结果产生前冻结。

### 训练数据

使用已有 GT 建立：

1. 常规正确关联。
2. 同一人物的跨摄像头关联。
3. 视觉相似但不同身份的困难负例。
4. 同场景存在多个竞争身份的情况。
5. 原始 GMT 已经正确的保护样本。
6. 原始 GMT 错误且正确历史身份合法可用的困难样本。

允许通过真实 TRAIN 外观向量构造受控污染历史，作为辅助鲁棒性实验。

但是：

合成污染不得当成天然真实纠错事件；
不得混入独立正式评价结果；
不得直接赋予未来安全价值；
不能通过 GT 在推理时挑选正确身份。

同一人的跨摄像头观测、相邻时刻与同一身份错误事件不能被简单当成彼此独立样本。

必须保留自然事件分布与困难事件的采样权重，避免大量复制极少数错误样本。

### 损失函数

优先保留既有匹配监督与合法联合分配损失。

新增：

Cross-Camera Identity Consistency Loss；
Identity Continuity / Stability Loss；

必要时加入当前合法完整分配的结构化排名损失。

未知身份和未认证负例必须显式 mask。

当前完整分配 GT 可以用来训练当前决定是否合理，但不能将此损失称为未来长期价值函数。

本阶段不训练 H32/H64 长期未来回报 Head，除非出现新的、真实合格且预算允许的原生监督。

### 训练策略

首先复用已训练的 Phase XIII/XIV 模型。

预注册一个小规模训练预算，例如：

Tiny：128 updates。

Pilot：1000–2000 updates。

正式本周对照：4000 updates以内，优先复用冻结主干，仅训练真正新增模块。

若需要继续训练更多层或使用其他预算，必须先得到小规模实验的合理证据，并让所有强对照获得公平预算。

随机种子优先使用既有：

20261008
20261009
20261010

禁止因为某一随机种子退化而仅报告较好种子。

---

## 7. 第五阶段：正式公平比较矩阵

必须至少包括：

M0：原始完整 GMT/GTA。

M1：同前端余弦关联。

M2：GMT + 固定阈值。

M3：GMT + 可学习阈值。

M4：GMT + 少量可学习融合权重。

M5：GMT + 原规则二次关联。

M6：GMT + MLP/DeepSets。

M7：GMT + Set Transformer / 普通 Attention。

M8：Phase XIII Fixed Question。

M9：Phase XIV Multi-question JEV。

M10：Phase XV V3 参考。

M11：本周新的 JEVTrack。

如果全部旧控制具有可靠、同条件、同 SHA 的可复用结果，则可以复用，不必为了形式重复大量训练。

如果历史结果存在不同视觉输入、不同GT处理、不同solver、不同训练预算或不同前端，则必须重新运行相应公平对照，不能拼接表格。

尤其注意：

原始 GMT GTA 使用的 VFCE/RPCE、ROI处理和关联路径与当前某些JEV缓存并不一致。

以前共同缓存的点式 GTA 实验不能冒充完整 GMT。

应当优先复现原始 GMT 的完整真实运行。

如果因权重、资源或输入契约缺失无法恢复，报告阻塞，并继续完成同前端强控制，不得虚构原始 GMT 主表。

### 核心消融

至少测试：

A0：最强普通关联基线。

A1：只增加跨摄像头一致性。

A2：只增加身份稳定性。

A3：两者同时使用。

A4：固定问题代替动态问题。

A5：移除共享状态。

A6：移除完整分配交互，仅保留逐边评分。

A7：相同额外求解次数，但不使用JEV。

A8：相同全部证据的普通Attention。

若 A3 没有比 A1/A2 更好，不能强行声称两项创新具有互补性。

若 A4 与 A3 等价，说明动态 JEV Question 的独立价值尚未成立。

---

## 8. 第六阶段：真实多摄像头在线评价

所有最终主结果必须来自完整 native closed-loop tracking。

不能只比较离线余弦排序、单条候选评分、历史预测重放或短窗口 oracle。

必须真实运行：

检测 / ReID
→ 当前 Global ID 关联
→ 各摄像头原生合法分配
→ Bank 恢复或新身份
→ Gallery 提交
→ 下一帧重新读取真实状态。

官方和已有评测工具保持冻结。

完整报告：

HOTA
DetA
AssA
IDF1
MOTA
IDSW
Frag
CVIDF1
CVMA

其中CVIDF1/CVMA使用仓库已验证的官方MATLAB实现。

跨摄像头部分还必须报告：

- 不同摄像头间正确Global ID关联。
- 跨摄像头错误身份合并。
- 身份碎片和错误跳转。
- Global ID 跨视角持续一致性。
- 基于实际预测ID的跨摄像头身份检索precision/recall。
- 错误历史引起的索引污染和错误返回。

对比 GMT 和 JEV 的真实纠错：

N01：原来错误，JEV修正正确。

N10：原来正确，JEV反而改错。

但N01/N10必须说明评估口径：

独立重放的逐行结果；
同一状态的局部反事实；
完整在线变异状态下的最终结果。

不同口径不得相互混用。

另外报告实际 ID 错误持续时间、已知错误owner、UNKNOWN和右删失事件。

不要把IDSW下降自动解释为身份纠错。

模型可以通过不断创建新ID减少一些错误统计，却引起严重碎片化，必须避免。

---

## 9. 第七阶段：参数、FLOPs、FPS专项实验

新增代码实现：

reproduction_tools/jev_oneweek_count_params.py

reproduction_tools/jev_oneweek_profile_latency.py

必须实际运行 PyTorch 模型统计，而不是只从论文或源码估算。

参数至少区分：

Total Loaded
Total Registered
Trainable
Active Forward
Dormant
Removed GTA
New JEV
Net System Change

推理成本至少区分：

Stage1图像读取
Backbone
Detector
VFCE
历史读取
共享状态编码
跨摄像头一致性Reader
身份稳定性Reader
方案生成
方案评价
Hungarian
Gallery/Bank更新
Stage2 total
End-to-End Scene FPS

全部在同GPU、同输入、同前端条件下比较。

测试：

GMT original；
Cosine；
Fixed Question；
Multi-question；
Phase XV V3；
New JEVTrack；

以及主要普通模型控制。

每种至少三次完整重启，报告实际值、平均值、波动与峰值显存。

记录GPU型号、并发负载、PyTorch/CUDA、warmup方式、真实camera payload和完整scene frames。

不能把两摄像头camera-payload FPS当作scene FPS。

不能把共享问题的摊销每问题时间当作单次跟踪延迟。

要明确：

增加 JEV 是增加网络，还是替换 GTA；
净参数是否减少；
总 FPS 是否下降；
Stage2 是否明显变慢；
如果出现精度增益，单位计算成本是否值得。

---

## 10. 科学验收与版本处理

提前冻结研究目标，不得结果出来后改口径。

建议将方法科学成功分成三级。

### 一级：功能与实验成立

新模块：
可训练；
无GT泄漏；
合法候选正确；
跨摄像头ID语义正确；
原生在线状态不被非法修改；
候选顺序和动态数量测试通过；
真实FPS可以测量。

这是工程通过，不代表论文成功。

### 二级：关联能力成立

相比最强同条件普通关联控制：

HOTA和AssA具有明确正向变化；
CVIDF1具有一致性收益；
IDSW没有不可接受的恶化；
正确身份保护能力得到证明；
困难跨摄像头情况不是只靠合成样本获得提升。

可以将HOTA比最强同条件控制高0.5个百分点作为探索性目标，但必须说明这不是统计显著性的保证。

### 三级：JEV独立价值成立

必须同时证明：

不是仅仅增加二次关联；
不是固定或可学习阈值就足够；
不是简单可学习融合或同证据MLP/Attention就能获得相同收益；
参数、延迟和FPS代价可接受；
多问题共享状态、动态问题及类型化决策确实发挥了作用。

五项有一项缺乏证据，应当明确标为NOT_ESTABLISHED，不能偷偷改成PASS。

特别是：

如果JEV优于GMT但与普通Attention相当，研究可能仍有工程价值，但不能宣称JEV架构独特优势。

如果JEV的最佳版本只是Fixed Question，论文不能把动态多问题机制写成必不可少的创新。

---

## 11. 遇到失败不能直接结束，但也不允许无限实验

建立单独的一周执行控制器：

reproduction_tools/jev_oneweek_controller.py

工作规则：

1. 当前实验失败，先确定是工程错误、数据不足、结构问题还是科学负结果。
2. 普通工程错误可以修复后重跑，保留原失败记录。
3. 科学假设失败，冻结结果，不重复相同设置碰运气。
4. 如果有明确的原因，可以执行下一条预注册研究假设。
5. 每次仅修改一个主要因素，确保能归因。
6. 至多进行三轮主要算法修改。
7. 先使用小规模验证，再决定是否扩大GPU训练。
8. 禁止通过降低Gate、重复访问DEV或挑选好种子制造成功。
9. 当本周研究预算耗尽，形成准确的最终报告和下一步建议。
10. 主MASTER GOAL不因为某阶段NO_GO而被标为已完成。

优先修复顺序：

历史证据不足；
跨摄像头身份信息未正确使用；
动态问题没有独立监督；
身份正确性与原生Global ID承诺不一致；
多检测联合分配引起的身份冲突；
推理实现效率不足。

不要在任何一个问题出现时直接转向更大Transformer或更长训练。

---

## 12. 七天工作安排

### Day 1 — 10月12日

完成历史资产和基线审计。

确认Phase XIII/XIV/XV全部必要代码及权重是否存在。

恢复一个可重复的原始GMT比较路径。

完成参数量核对。

完成严格TRAIN/开发集数据范围冻结。

形成五个审稿人问题对应的实验清单。

首次提交GitHub。

### Day 2 — 10月13日

完成新JEVTrack数据与监督接口。

构造跨摄像头同一性、历史稳定性和合法分配的训练事件。

统计真实困难样本、UNKNOWN和标签可辨识性。

验证候选构造的合法性、动态数量和跨摄像头身份语义。

完成第二次提交。

### Day 3 — 10月14日

实现基于Phase XIV的新模块。

复用旧模型初始化，保持初始输出等价。

完成Tiny训练、单元测试、梯度测试、候选置换和原生提交测试。

修复工程错误。

完成第三次提交。

### Day 4 — 10月15日

执行小规模Pilot。

复用或训练普通融合、动态阈值、二次关联、MLP、DeepSets、Set Transformer等必要对照。

分析：
JEV实际新增纠错；
反改错；
跨摄像头一致性变化。

只在有证据时调整一项设计。

完成第四次提交。

### Day 5 — 10月16日

完成开发集完整真实在线追踪。

输出：

HOTA / AssA / IDF1 / IDSW / CVIDF1 / CVMA；
跨相机检索；
原生错误传播；
N01/N10。

任何完整视频身份崩溃都要记录，并恢复安全基线。

完成第五次提交。

### Day 6 — 10月17日

完成必须的消融、至少一组同条件完整GMT效率对照，以及参数量/FPS分析。

若有正向结果，补齐其他随机种子。

若新JEV仍明显不如普通Attention或Fixed，不再继续堆模块。

形成论文主表草稿。

完成第六次提交。

### Day 7 — 10月18日

冻结所有已有的真实结果。

更新研究贡献判断。

完成：

算法结构说明；
实验主表；
消融表；
参数量/FPS表；
五个审稿人问题答复；
失败与局限；
WWW研究相关性；
摘要是否具备投稿依据的判断。

WWW的Web相关性要具体落在持续更新的多摄像头身份索引及检索可靠性上，不得只因为研究使用网络视频，就宣称符合WWW范围。

最终推送GitHub，核实远端提交一致。

---

## 13. GitHub提交要求

每完成一个阶段必须：

检查代码可运行；
运行相关测试；
保存配置；
保存原始指标；
保存失败情况；
保存模型与数据 SHA；
更新研究Goal；
git commit；
git push。

只上传：

源代码；
配置；
必要实验结果；
小体积指标；
说明文档；
论文表格；
复现命令与清单。

原始视频、完整图像、数GB权重、原生轨迹和大型缓存仍保留在服务器。

不得强行push，不得修改历史阶段的冻结结果。

新建：

docs/JEV_ONEWEEK/RESEARCH_GOAL.md

docs/JEV_ONEWEEK/HISTORICAL_EVIDENCE_AUDIT.md

docs/JEV_ONEWEEK/METHOD_AND_INNOVATIONS.md

docs/JEV_ONEWEEK/REVIEWER_FIVE_QUESTIONS.md

docs/JEV_ONEWEEK/BASELINE_PARITY.md

docs/JEV_ONEWEEK/FINAL_REPORT.md

reports/JEV_ONEWEEK/EXPERIMENT_REGISTRY.json

reports/JEV_ONEWEEK/MAIN_RESULTS.json

reports/JEV_ONEWEEK/ABLATIONS.json

reports/JEV_ONEWEEK/PARAMETERS_AND_FPS.json

reports/JEV_ONEWEEK/CROSS_CAMERA_ANALYSIS.json

reports/JEV_ONEWEEK/GO_NO_GO.json

reports/JEV_ONEWEEK/ARTIFACT_SHA_MANIFEST.json

任何没有运行的实验必须明确标记 NOT_RUN，说明原因，不能用零代替。

---

## 14. 最终报告必须回答的十二个问题

1. 原始GMT与现有最佳JEV分别达到什么水平？
2. 新模型是否真正提高多摄像头Global ID一致性？
3. 哪些身份错误是JEV真正纠正的？
4. 哪些原来正确的关联被JEV改错？
5. 比简单二次关联多获得了什么？
6. 比固定及可学习阈值多获得了什么？
7. 比少量可学习融合参数、MLP和普通Attention多获得了什么？
8. 为什么新的两个问题需要共享状态，并且不能退化成一个普通分类Head？
9. 当前实现是否符合State–Question–Option–Decision的JEV思想？
10. 总参数、新增参数、净参数、FPS、Stage2延迟和显存到底是多少？
11. 是否在真实在线多摄像头跟踪中提高了HOTA、AssA、CVIDF1，并保持身份稳定性？
12. 现有结果是否足以支撑WWW 2027的一篇具有明确独立贡献的论文？

必须给每个问题列出：

结论；
支持它的实验；
对应模型/权重；
具体数值；
复现路径；
局限；
证据状态。

---

## 15. 最终执行指令

立即开始。

本次不要再把主研究目标写成“学习候选价值”或“在多个方案中做选择”。

这些是内部计算步骤。

我们真正研究的是：

**如何在多摄像头多目标跟踪中，利用跨摄像头证据提高Global ID关联的准确性，并保持长期身份稳定。**

JEV只是我们提出的实现方式。

必须用公平实验回答为什么需要它。

不要继续无目的地增加Question、Head、阈值或网络规模。

优先继承已经跑通的Phase XIV/XV代码、权重、native执行器和已有实验数据。

第一优先级是恢复基线并完成同条件比较。

第二优先级是验证跨摄像头一致性模块。

第三优先级是验证身份稳定性模块。

第四优先级是回答五个审稿人问题并完成参数与FPS实验。

如果模型有效，完成完整的论文证据链。

如果模型无效，也必须完成有限的原因分析及强基线对照，而不是第一次NO_GO就停止研究。

所有科研结论必须基于真实训练和真实在线实验。

最终论文能否声称JEV具有独立贡献，由实验决定。

**START NOW.**