
# FINAL MASTER GOAL — WWW / JEV-MCMOT
# Phase XVII and Subsequent Evidence-Driven Research

项目：JEV-MCMOT
论文目标会议：The Web Conference (WWW 2027)
GitHub：https://github.com/LYQ1107/WWW

当前冻结研究起点：
Phase XVI commit:
666e2b922eb676b3552db0c6dc8b67b1b3978450

当前 Phase XVI 本地项目：
/home/liuyeqiang/WWW_jev_phase16

建议新项目：
/home/liuyeqiang/WWW_jev_phase17

建议新分支：
jev/www-jev-phase17-identity-evidence-20261011

你的角色：
你是本项目的首席研究工程师、实验执行者与科学证据审核者。你必须持续维护一个明确的长期科研 Goal，而不是仅完成单个实验阶段。

========================================
0. 创建永久的 MASTER GOAL
========================================

首先在新研究分支中建立：

docs/WWW_JEV_MASTER_GOAL.md
reports/JEV_MASTER/MASTER_GOAL.json
docs/WWW_JEV_RESEARCH_ROADMAP.md
docs/WWW_JEV_PAPER_READINESS.md
docs/PHASE17_RESEARCH_GOAL.md

MASTER GOAL 名称：

Evidence-Grounded Identity Lifecycle Decision Learning
for Reliable Online Multi-Camera Tracking

论文暂定名称：

JEV-MCMOT:
Evidence-Grounded Identity Lifecycle Decisions
for Online Multi-Camera Tracking

最终科学问题：

在在线多摄像头多目标跟踪中，如何利用真实可观察的视觉历史和时空证据，可靠地判断 Global ID 的归属、选择当前身份、维护身份记忆，并在目标消失后恢复身份，从而减少长期身份错误传播？

最终希望建立一个具有以下能力的统一决策模型：

A. MATCH / WHO
对当前检测与具体合法 Global ID 候选进行决策，纠正错误匹配，同时尽可能保留正确关联。

B. REACTIVATION / OLD OR NEW
在真实 stale-bank 提供合法旧身份时，判断恢复旧身份还是创建新身份。

C. MEMORY / WRITE
在最终身份提交后，判断哪些真实历史观测应影响在线身份表征。

D. Identity Ownership Reliability
贯穿上述三个任务，估计历史身份归属及其证据的可靠性，特别处理多 ID 别名、混合身份历史、证据缺失及不确定状态。

E. Shared Visual Decision Computation
借鉴 Visual Jev 的共享视觉计算、Question/Option 分支执行和现有 LM head 概率读出，并测试是否能够在我们的 MCMOT 中获得额外收益。

最终模型必须满足：

1. 严格在线、因果推理。
2. GMT 感知作为可冻结、可复现的基础。
3. 不使用未来帧或 GT 进行运行时决策。
4. 决策实际影响 native identity commit 和后续帧。
5. 相比 GMT 以及强固定规则具有可信收益。
6. 与 MLP、DeepSets、Attention 等强模型公平比较。
7. 必须证明 JEV 决策机制的独立价值，或明确将贡献归因于其他有效机制。
8. 身份错误传播、跨摄像头关联与长期身份检索质量改善。
9. 达到明确的在线推理与部署性能要求。
10. 最终形成符合 WWW 研究范围的可复现论文。

注意：
这是科学目标，不是预先成立的研究结论。

不允许将“代码完成”“完成全部阶段”或“模型经过训练”解释为 MASTER GOAL 科研成功。

MASTER_GOAL.json 必须包含：
- goal_id
- final_scientific_objective
- target_conference
- frozen_reference
- scientific_success_criteria
- web_relevance
- permitted_data_scope
- forbidden_data_scope
- current_phase
- current_phase_status
- master_status
- active_hypotheses
- rejected_hypotheses
- evidence_registry
- artifact_registry
- next_experiment
- remaining_authorized_budget
- conditions_for_final_evaluation
- paper_readiness

MASTER 状态至少支持：
ACTIVE
SCIENTIFIC_GO
SCIENTIFIC_NO_GO
BLOCKED_EVIDENCE
BLOCKED_ENGINEERING
PIVOT_REQUIRED
READY_FOR_FINAL_EVALUATION
SUBMISSION_READY

某个 Phase 可以完成并标记 NO_GO，但不得由此自动将长期 MASTER GOAL 标记为 COMPLETE。

如果某个科研假设失败，保留其结论；在新版本中选择有实验证据支持的其他假设继续研究。

不允许无限调整同一个模型寻找正向数字。

========================================
1. 永久保护历史研究证据
========================================

首先核对 Phase V–XVI 历史分支、最终报告、提交及实际 checkpoint。

重点保护：

Phase XV 冻结参考：
HOTA 74.188270
AssA 74.890708
IDF1 90.521042
IDSW 147
CVIDF1 90.285890
CVMA 86.781450

Phase XVI 安全记忆诊断：
HOTA 61.764421
AssA 52.052180
IDF1 71.026365
IDSW 1494
CVIDF1 67.476069

Phase XVI：
- 96 条原生变异因果分支
- 18 个严格有害 prefix/H 检查
- 2 个独立安全收益 cluster
- 0 个合格安全恢复标签 cluster
- 真实图像 FPS 约 3.25
- Stage2 p95 约 56 ms
- NO_GO_SAFE_MEMORY
- NO_GO_RECOVERY_LABEL_ELIGIBILITY

这些负结果不允许覆盖、删除或重新解释成 PASS。

注意：
Phase XV 的正向指标属于已经多次使用的 DEV17/18/19 上的诊断，不是独立测试。

Phase XVI 的安全记忆退化来自冻结权重下的机制变化，不能表述为新 JEV 模型训练失败。

Phase XVI 中 H8/H16/H32 的重叠窗口，不得当成独立实验样本。

保留：
- 历史 checkpoint
- 配置及 SHA
- 全部原始预测
- 原生状态
- 原生因果分支
- 失败日志
- MATLAB 评测版本
- 原始研究报告

从固定提交建立独立 worktree。
不修改已有分支历史。
不 force push。
不使用破坏性 reset。
不删除任何未证明安全可删除的科研文件。

全部新实验使用独立 run ID、独立输出目录及完整 provenance。

========================================
2. PHASE XVII-A：系统学习开源 JEV
========================================

先研究开源模型，不要直接修改 GMT。

重点审计六个仓库。

A1. Visual Jev

https://github.com/guanxuyu-sv/Visual-Jev

重点阅读：
README.md
REPRODUCE.md
code/vdm/models/vdm_model.py
code/vdm/models/prompts.py
code/vdm/training/train.py
code/vdm/training/losses.py
code/examples/quickstart.py

逐函数核查：

VDM.prepare_group
VDM._prefill_prefix
VDM._fork
VDM.run_prefix_share
VDM.run_prefix_share_batch
VDM._readout

必须回答：

1. 共享视觉前缀具体包含哪些信息？
2. 独立问题的 suffix 如何形成？
3. KV cache 怎样克隆与复用？
4. M-RoPE 位置编码如何保持一致？
5. 如何从原 LM head 读取候选概率？
6. 为什么不需要额外训练 linear decision head？
7. 这里的“不需要新 head”为什么不等于“不需要模型训练”？
8. 推理共享是否能够保持 full forward 的数值与决策一致？
9. N=1、4、8、16、32 时真实总延迟和摊销延迟有什么区别？
10. 在动态身份候选超过16个时，如何避免候选截断或选项槽位偏差？

必须复核：
无新增 head 的预训练版本
无新增 head 的 LoRA 版本
有专用 head 的对照版本

禁止将 Visual Jev 作者的吞吐数字直接作为 GMT 在线跟踪延迟。

A2. Valen

https://github.com/Liuziyu77/Valen

重点阅读：
valen/modeling/dual_encoder/model.py
valen/modeling/dual_encoder/layers.py
valen/modeling/qwen/heads.py
valen/modeling/qwen/batching.py
valen/training/rlcd.py

重点研究：
- Shared State Memory
- QuestionBlock
- CandidateBlock
- 动态候选表示
- 候选证据读取
- Choice / Noul / Score
- SFT 和实验性 RLCD

分析如何改造成 MCMOT 中的：
WHO
OWNER RELIABILITY
WRITE
OLD OR NEW

不得声称复制了 TypeSafe 未公开的商业 Jev 架构。

A3. OneJev

https://github.com/OmniJev/OneJev

重点阅读：
qev/engine.py
qev/mm_engine.py
qev/prompt.py
qev/schema.py
qev/calibrate.py

学习：
- LM head 选项读出
- shared-prefix
- batch cache fork
- 动态候选
- 独立拒答
- 概率校准
- 大候选集合分块

特别检查：
Qwen3.5 的缓存可能同时包含 attention 和 recurrent/conv states。

不能只复制注意力 KV cache 而遗漏其他状态。

A4. Jev-Mem

https://github.com/libingzheren/Jev-Mem

阅读：
memory/jev_mem_retrieval.py
memory/jev_mem_policies.py
memory/jev_questions.py
memory/jev_client.py

分析：
原始观测存储
检索候选生成
候选质量评分
停止/继续判断
检索预算控制

重点借鉴：
原始历史记录与实际参与关联的历史证据可以分离管理。

A5. NanoJev

https://github.com/TianyuCodings/NanoJev

研究：
动态候选评分
多问题共享结构
小模型
概率输出
控制任务训练
动作决策与实际环境状态反馈

区分结构化游戏决策与真实视频身份跟踪的差异。

A6. vLLM-Jev

https://github.com/mode-io/vllm-jev

研究：
动态批处理
缓存复用
跨请求缓存安全
并发一致性
推理延迟
显存开销
冷启动和warmup

只借鉴可验证的执行与部署机制，不因更快而默认科学表现更好。

每个外部仓库必须记录：
Git commit SHA
许可证
实际阅读的代码路径
核心函数
输入输出
推理机制
训练方式
可复用部分
不可移植部分
在 GMT 中的具体适配建议

建立：
docs/PHASE17_VISUAL_JEV_SOURCE_AUDIT.md
docs/PHASE17_EXTERNAL_ARCHITECTURE_MATRIX.md
reports/JEV_PHASE17/EXTERNAL_CODE_MANIFEST.json

要求源码级分析，不得只复制 README。

========================================
3. PHASE XVII-B：复现安全记忆崩溃的根因
========================================

这是本轮最高优先级。

不要训练任何新 JEV。

冻结 Phase XV 的同一个权重和 GMT Stage1。

逐函数阅读：

gtr/modeling/jev_phase16/safe_memory_admission.py
gtr/modeling/jev_phase16/safe_matching_memory.py
gtr/modeling/jev_phase16/native_memory_commit_adapter.py
gtr/modeling/jev_phase15/native_commit_adapter.py
gtr/modeling/jev_phase15/commitment_option_reader.py
gtr/modeling/meta_arch/gtr_rcnn.py

目标：

严格确定为什么引入 bounded_quarantine 后，
HOTA 从74.188下降到61.764，
IDSW从147上升到1494。

建立完整的 2×2 因果实验：

B0：Original WHO + Original Q4 Memory
B1：Safe WHO + Original Q4 Memory
B2：Original WHO + Safe Q4 Memory
B3：Safe WHO + Safe Q4 Memory

所有实验必须：
使用同一权重
使用相同外生视觉输入
使用相同原始候选规则
使用完整 native online commits
在分支内部真实更新后续状态
不复用原始 GMT 的未来动作
不使用 GT 进行推理
记录完整原始预测
使用相同的官方评测器

分别研究：

1. 从冷启动完整视频运行。
2. 从相同历史 prefix 开始运行。
3. 当前决定发生变化的第一次事件。
4. 首次错误变化如何影响以后帧。
5. 不同记忆策略是否改变 Q4 输入分布。
6. 不同策略是否触发不可信 fallback。
7. 是否减少了正确历史的有效覆盖。
8. 是否强化了已经错误的 Global ID。
9. 哪些决策引起 false merge。
10. 哪些决策引起 false split 或 wrong birth。

对每条发生状态分歧的路径保留：

frame / camera / detection
original ID
new ID
candidate scores
matching history
raw Gallery
trusted Gallery
Q4 input
candidate reliability
fallback reason
first-divergence fingerprint
subsequent wrong-ID episodes

必须区分：

真正的原生因果变化
相关但非因果的日志差异
已知 GT 错误
UNKNOWN owner
合法身份别名
同一预测 ID 中的混合人物历史

输出：
PHASE17_FIRST_DIVERGENCE_AUDIT.json
PHASE17_WHO_Q4_FACTORIAL.json
PHASE17_COLD_START_VS_PREFIX.json
PHASE17_MEMORY_FAILURE_MECHANISM.md

在尚不能解释原因时，不得直接扩大训练。

但普通工程缺陷应当主动修复并重新验证，不应因为脚本首次失败就结束整个 Goal。

========================================
4. PHASE XVII-C：身份归属的可辨识性研究
========================================

研究核心：

Appearance consistency
不等于
Global identity ownership correctness。

基于真实历史事件建立 Owner Evidence Audit。

将身份历史分为：

C1. 过去证据足以明确区分正确身份。
C2. 正确人物出现过，但当前 Global ID 归属存在别名或混合。
C3. 正确人物历史存在，但候选生成器没有提供合法身份。
C4. 现有过去证据无法区分两个或多个竞争身份。
C5. 当前观察证据确实缺失。
C6. 部分历史能够识别人物，但缺乏可靠身份归属。

注意：
这些状态是离线诊断类别，不得把离线 GT 诊断类别直接作为运行时模型特征。

建立过去可观察的证据：

当前真实 ReID 特征
过去真实 ReID 特征
不同时间尺度的视觉一致性
same-camera appearance
cross-camera appearance
track age
motion compatibility
temporal consistency
camera consistency
identity candidate margin
Global ID history stability
historical disagreement
available stale-bank identity

不允许使用：
未来帧
未来GT
真实GT ID数值
未在当前原生候选池中的oracle身份
事后纠错结果

GT只用于离线标注、分类和评价。

评估：

correct-owner candidate availability
alias frequency
mixed-owner history rate
owner separability
known/unknown coverage
confirmed wrong owner rate
correction opportunity
false correction opportunity
cross-camera identity consistency

重点回答：

现在的模型究竟是选错了已存在的正确身份，
还是根本没有足够信息确定正确身份？

如果当前可用证据已经不能区分不同人物，
必须明确记录该信息上限。

不得声称增加训练次数能够解决信息不存在的问题。

建立独立的、可追溯的训练事件组。

所有训练事件组必须有：
source video
time block
GT-free event key
identity alias grouping
conflict group
sampling probability
known/unknown mask
valid counterfactual outcome
provenance hash

禁止将同一错误事件相邻几十帧当成几十个独立纠错样本。

输出：
PHASE17_OWNER_EVIDENCE_AUDIT.json
PHASE17_IDENTIFIABILITY_REPORT.md
PHASE17_CORRECTION_OPPORTUNITY.json

========================================
5. PHASE XVII-D：无新增 Linear Head 的 Visual JEV 对照
========================================

这是对 Visual Jev 最直接的借鉴。

目标：
判断预训练视觉语言模型利用原 LM head，
能否在真实困难身份事件中产生新的可靠区分证据。

先建立独立 OFFLINE TEACHER。

原则：
不要直接将4B VLM部署到GMT逐帧路径。
先在训练数据中的真实困难身份事件上做有界实验。

使用过去与当前实际存在的视觉图像和身份历史。

构建视觉问题：

WHO:
当前观测是否与候选历史身份一致？

OWNER:
这组历史观测是否提供一致、可辨识的身份归属证据？

AMBIGUITY:
当前证据是否不足以可靠地区分这些候选？

CROSS-VIEW:
历史跨摄像头观测是否与当前目标兼容？

候选只能来自真实可用历史和合法状态。

若需把多个历史外观裁剪组织为一张对比图，
必须使用实际过去和当前图像，记录来源、时间、视角与图像处理。
不得加入未来帧或人工正确答案标识。

先比较：

D0：GMT原始视觉关联分数
D1：普通视觉余弦与时序规则
D2：未经任务适配的Visual Jev式 LM head选择
D3：有明确许可和必要数据时的LoRA任务适配
D4：使用相同视觉证据的MLP/DeepSets/Attention

D2不得额外训练新的linear decision head。

不能将视觉语言模型因训练数据、额外图像或额外计算预算获得的优势，归因于JEV特殊结构。

必须做：
独立问题的full forward parity
shared-prefix forward parity
batch执行一致性
不同候选顺序一致性
不同候选数量稳定性
已知假前提拒答
真实 UNKNOWN case
真实同人/异人判别
多ID别名分析
混合历史分析
置信度校准

Visual Jev的原始演示支持2–16个选项。
我们的身份候选数可能明显超过16。

不能直接Top16截断，然后假装正确身份一直在候选池里。

首先计算候选 Recall@K。
对于更多候选，应研究合法的分层检索与分块评分，并验证分块间比较是否可校准。

不得直接比较来自不同独立softmax集合的概率大小。

动态身份候选的排列变化不得改变语义身份判断。

还必须测试：
同一时刻不同问题是否能够共享公共前缀。
状态提交后是否正确失效缓存。
不同frame、camera和identity history不能误用旧缓存。
独立问题之间不能泄漏其他问题的答案。

最终回答：

是否存在原GMT+历史数值证据无法区分，
但视觉JEV能够区分的真实身份错误？

如果没有，必须记录 NEGATIVE RESULT。

不要为了使用Visual Jev而继续引入更大的模型。

输出：
PHASE17_VISUAL_JEV_TEACHER.md
PHASE17_NO_NEW_HEAD_COMPARISON.json
PHASE17_VISUAL_EVIDENCE_VALUE.json
PHASE17_SHARED_PREFIX_PARITY.json

========================================
6. PHASE XVII-E：设计可靠但不破坏历史的记忆机制
========================================

只有完成身份信息和失败因果审计，
才能设计新的记忆策略。

不要继续默认使用Phase XVI的hard quarantine。

首先测试低风险替代：

E0：Phase XV原始记忆
E1：Raw Gallery完整保留，采用软可靠性权重
E2：时间与跨相机条件化历史权重
E3：存在足够证据时的Owner-aware Memory
E4：条件允许时的Visual-JEV-assisted evidence weighting

要求：
- 所有真实原始观测仍可追溯。
- 不删除已存在Global ID。
- 不把低相似度直接等价于错误身份。
- 不将三帧一致直接等价于身份正确。
- 没有可靠证据时执行明确的保守fallback。
- 不把ABSTAIN与START_NEW混为一谈。
- 不把DEFER直接当成安全恢复动作。
- 不自动创建原生Bank没有提供的旧身份。
- 所有记忆权重只由当前和过去信息计算。

重要：
Memory Evidence Reliability
与
Identity Owner Reliability
必须分开建模。

一组历史向量彼此相似，并不证明该Global ID归属正确。

可考虑：
基于多个历史分组的owner consistency
跨视角一致性
历史污染风险
原始匹配置信度
必要纠错证据
状态不确定性

优先将可靠性用于证据排序或加权，而不是永久删除历史。

每个新策略必须经过：
无GT在线输入检查
原始状态完整性检查
WHO/Q4输入对比
小规模原生因果反事实
完整冷启动诊断
反改错及误出生检查

如果策略在完整视频上造成大规模身份崩溃，立即停止该策略的资格升级，保存失败数据，恢复冻结参考。

不得在失败策略上继续正式20k训练。

========================================
7. PHASE XVII-F：候选级JEV与长期决策训练
========================================

只有前面的证据和安全条件通过，
才进入正式学习式决策研究。

模型设计优先考虑：

Shared Visual Identity Context
Dynamic Typed Question Reader
Question-conditioned Candidate Reader
Owner Reliability Representation
Legal Option Readout
Consequence Value Estimation
Native Assignment / Commit

必须测试两种不同路线：

F1. No-new-head visual LM readout
参考Visual Jev。
使用已有LM head读候选概率。
作为视觉决策教师或可部署候选。

F2. Lightweight native JEV controller
利用GMT已有ReID特征、轨迹历史和证据。
使用轻量typed reader，满足在线延迟要求。

不能预设F1或F2更优。
若F1只适合作为离线teacher，可进行严格受控的知识蒸馏，但需要独立报告蒸馏收益和额外监督成本。

MATCH使用：
合法active candidates
真实候选证据
共享全局assignment

REACTIVATION使用：
原生Bank提供的合法stale IDs
真实记忆读取
明确的NEW与保守回退语义

MEMORY使用：
最终身份归属
实际历史观测
可靠性权重与可执行更新动作

决策顺序必须遵守原生GMT：
当前匹配
→ 必要的旧身份恢复
→ 最终身份提交
→ 历史更新
→ 后续帧

独立候选可以并行评分。
依赖状态提交的任务不能错误并行。

正式训练前必须具备：
足够独立的有效训练事件组
正确身份与错误身份的真实监督
真实反改错负例
合法动作集合
原生状态转移一致性
已执行分支的因果价值
UNKNOWN和censored mask
可靠的验证分割

必须将Identity Correctness
与Long-Horizon Action Utility分开。

不允许：
把UNKNOWN当成错误
把未执行分支效用设为0
把局部edge utility直接视为联合Hungarian assignment的精确价值
把实际未来GT作为运行时特征

先运行Tiny sanity。
然后小规模pilot。
满足预注册资格后，再决定是否运行正式训练。

========================================
8. PHASE XVII-G：必须完成的公平实验
========================================

如果进入正式模型比较，至少包含：

G0 GMT original
G1 固定关联规则
G2 Learned threshold
G3 Candidate MLP
G4 Candidate DeepSets
G5 Candidate Set Transformer
G6 Phase XV frozen JEV
G7 Visual-JEV no-new-head
G8 Lightweight evidence-conditioned JEV
G9 MATCH + validated MEMORY
G10 MATCH + validated REACTIVATION
G11 Unified lifecycle JEV（仅满足资格后）

公平要求：
相同候选池
相同合法动作
相同原生assignment约束
相同可使用的视觉证据
相同训练数据
相同正确性和效用监督
相同模型选择流程
参数量与计算预算对照
完整运行时成本披露

新增视觉输入时必须增加：
有/无视觉输入消融。

新增长期因果监督时必须增加：
有/无该监督消融。

新增共享问题结构时必须增加：
共享JEV与独立普通网络的对照。

不能只用GMT OFF与JEV比较，就宣称JEV网络结构具有独立优势。

如果普通MLP、Attention或固定规则取得最好结果，必须如实报告。

========================================
9. PHASE XVII-H：真实在线安全与性能
========================================

对于所有获得pilot资格的方法，
必须进行真正的mutated-state在线闭环。

报告：
HOTA
AssA
IDF1
MOTA
IDSW
Frag
CVIDF1
CVMA

官方CVIDF1/CVMA使用已验证的MATLAB评测流程。

同时报告：
Owner correction N01/N10
corrected wrong identities
broken correct identities
false merge
false birth
false split
wrong-ID duration
memory contamination
candidate miss
unknown exposure
safe correction coverage
cross-camera identity consistency

对完整视频必须报告逐视频和真实pooled指标。
不能使用简单均值冒充pooled TrackEval。

运行时间必须分开：
图像读取
Stage1
视觉特征提取
JEV forward
prefix/cache reuse
candidate preparation
assignment
native commit
memory update
Stage2 total
end-to-end scene FPS

必须有：
single question latency
batched question latency
p50/p95
actual total batch time
amortized time/question
GPU memory
cache construction overhead

不能把Visual Jev的5.7ms平均摊销时间当成单次MOT决策延迟。

在线部署仍以达到真实双摄像头实时处理为最终目标，原先25 FPS和Stage2 p95门槛作为需要对照的既有目标；新增策略开销也须独立报告。

若原始感知管线已无法达到目标，
单独说明Stage1瓶颈。
不得为了让JEV看起来达到25FPS而排除真实图像处理。

性能优化不允许改变跟踪实验的视觉证据、候选或决策语义而不重新验证。

========================================
10. WWW科学贡献与身份检索
========================================

我们希望保持纯计算机视觉的核心任务。

但论文必须明确回答Web相关问题：

在持续输入的多源视频服务中，
如何维护一个可信、可查询的跨摄像头身份索引？

基于实际在线跟踪结果建立identity-index retrieval evaluation。

要求至少评价：
跨摄像头身份查询准确率
错误身份合并造成的查询污染
单个真实身份的跨视角连续性
身份错误持续时间
检索结果中错误轨迹片段比例
身份检索延迟
在线更新开销

该实验应建立在真实预测和可审计GT评价之上。

不要求额外开展带宽仿真或通信协议模拟。

不能因为使用了VisionTrack就声称具有Web贡献。

文章第一页面向WWW明确说明：
在线多媒体流中的跨摄像头身份索引可靠性，是一个具体的Web多媒体检索与持续更新问题。

同时研究：
JEV是否减少由错误Global ID引发的长期检索污染。

========================================
11. 预注册Gate与科学真实性
========================================

所有新实验必须事先冻结：
主要指标
次要指标
数据分割
模型选择规则
独立统计单位
有效训练预算
安全失败条件
允许的模型版本数
运行时要求

特别注意：

Phase XVI的原始positive-support召回已经达到99.389%。

不能继续把新增2个百分点的召回提升作为一个数学上不可能通过的新研究门槛。

原历史Gate不能回头修改。
新阶段必须依据剩余可提升空间、困难样本覆盖和身份安全性，重新预注册合理的科学问题与指标。

真正的安全硬门槛必须保留：
原生状态有效
无GT/future泄漏
无大规模身份崩溃
不隐瞒严重负迁移
不覆盖旧结果
不伪造监督
不将相关事件当作独立样本

对每个主要方法分别输出：
ENGINEERING_PASS/FAIL
DATA_ELIGIBILITY
SCIENTIFIC_GO/NO_GO
DEPLOYMENT_PASS/FAIL
INDEPENDENT_VALUE_ESTABLISHED/NOT_ESTABLISHED

允许一个机制实验失败后转向新假设。
不允许失败模型继续无条件启动正式大规模训练。

========================================
12. 持续研究与自动推进机制
========================================

这是MASTER GOAL的重要要求。

不要把“Phase XVII报告已经生成”当成整个任务结束。

创建一个可恢复的研究控制器：

reproduction_tools/jev_master_goal_controller.py

以及：
reports/JEV_MASTER/EXPERIMENT_REGISTRY.json
reports/JEV_MASTER/EVIDENCE_REGISTRY.json
reports/JEV_MASTER/NEXT_ACTION.json
reports/JEV_MASTER/RESEARCH_DECISIONS.md

每轮应：

1. 读取MASTER GOAL。
2. 检查已完成实验和原始证据。
3. 检查当前科学假设。
4. 判断该假设支持、失败还是证据不足。
5. 按照已经冻结的实验顺序执行下一项合格任务。
6. 自动修复可验证的普通工程错误。
7. 保留所有失败及原始输出。
8. 更新Goal及论文就绪状态。
9. 如果某机制失败，选择有证据支持的另一研究分支。
10. 继续推进，直到本次授权范围内的下一阶段研究任务完成。

允许的研究转向示例：

如果hard memory filtering导致崩溃：
转入soft evidence weighting。

如果soft evidence weighting同样失败：
审计Global ID ownership证据。

如果当前证据不能区分正确身份：
研究候选生成、跨摄像头时空约束或Stage1 ReID可分性。

如果Visual Jev没有提供额外身份信息：
不继续盲目扩大VLM容量，转入轻量原生模型。

如果JEV与Attention效果相同：
研究共享生命周期状态和真实行动价值是否有增量，不宣称架构独特。

但每条新假设必须：
记录明确动机
有独立验证目标
有受限预算
有否决条件
不违反sealed split
不无限调参

每一轮最多测试三个事前区分的主要研究假设。
预算或可辨识性耗尽时，报告PIVOT_REQUIRED或BLOCKED_EVIDENCE，并列出下一项需要的额外资源或证据，不得无限尝试。

禁止为了让MASTER GOAL显示SCIENTIFIC_GO而制造实验成功。

========================================
13. 数据授权与计算资源
========================================

本阶段继续沿用：

TRAIN：
video12
video13
video14
video16

反复使用的DEV：
video17
video18
video19

Controller heldout：
video20
video21
video22

official TEST：
禁止使用。

Full24：
本阶段不自动授权。

video20/21/22必须保持封存，直到独立的训练与验证Gate通过，模型和评价协议冻结，并满足既定解封条件。

不得提前读取其视频缓存、标签或正式评价结果进行调参。

Stage1可能已经接触全部24个TRAIN视频，因此controller-heldout不得宣称为完全未见过的全系统测试。

如果需要额外训练视频或新的独立数据集，
先输出研究必要性、预注册划分与资源影响，不得擅自污染最终测试集。

复用：
已有GMT checkpoint
已有VisionTrack
已有ROI/ReID cache
已有完整native状态
已有MATLAB评测环境

Visual Jev需要的新环境必须独立建立。
不允许为安装Qwen相关依赖而破坏当前GMT环境。

先检查：
nvidia-smi
可用显存
存储空间
网络状况
已有模型缓存
相关GPU任务

优先只在真实必要时下载外部权重。
大型4B/8B模型不得默认常驻逐帧生产系统。
不能抢占其他任务GPU。

遵守已有服务器存储限制，优先复用缓存和分块记录。

大模型、原始图像、完整状态、cache和checkpoint留在服务器。
GitHub只提交必要源码、配置、指标、manifest和研究报告。

========================================
14. WWW 2027 投稿进度
========================================

当前日期按2026-10-11执行规划。

WWW 2027官方时间：
摘要：2026-10-18 AoE
全文：2026-10-25 AoE

目标是尽可能形成有真实证据支撑的WWW研究论文，而不是仅完成工程系统。

立即建立：
docs/WWW_2027_SUBMISSION_READINESS.md

报告：
科学贡献是否成立
JEV是否有独立价值
当前最好真实基线
正式实验还缺什么
Web相关性是否具体
论文主要表格是否齐全
效率是否合格
哪些指标只有DEV证据
哪些实验尚未运行
是否已经达到诚实投稿的最低证据要求

如果截至投稿决策时尚无可靠科研增益，
不能编造论文实验数据，也不能将开发集结果包装为正式测试。

保留下一投稿周期的科学研究目标与全部证据。

========================================
15. 本次任务的实际执行顺序
========================================

从Phase XVI冻结状态开始：

P0：
创建MASTER GOAL、研究控制器及新的worktree。

P1：
完成Visual Jev、Valen、OneJev、Jev-Mem、NanoJev、vLLM-Jev源码审计。

P2：
复现WHO/Q4记忆反馈的2×2原生因果实验。

P3：
定位完整冷启动的第一批身份错误传播事件。

P4：
完成Owner Evidence可辨识性审计。

P5：
完成Visual Jev no-new-head有界视觉身份实验。

P6：
依据前述证据设计并比较软可靠性记忆策略。

P7：
满足监督和原生安全资格后训练轻量Evidence-Conditioned JEV。

P8：
完成相同证据、监督、求解器和计算预算下的强基线对比。

P9：
完成独立的真实在线验证和WWW身份检索评价。

P10：
判断最终是否具备Unified MATCH-MEMORY-REACTIVATION的科学证据。

P11：
更新论文就绪状态和下一阶段MASTER GOAL。

普通工程问题应自动修复并继续执行。
科学资格不满足时不运行依赖它的训练，但应完成不依赖该资格的其他已授权诊断任务。

========================================
16. 本次最终交付
========================================

必须创建并提交：

docs/WWW_JEV_MASTER_GOAL.md
docs/WWW_JEV_RESEARCH_ROADMAP.md
docs/WWW_JEV_PAPER_READINESS.md
docs/PHASE17_RESEARCH_GOAL.md
docs/PHASE17_VISUAL_JEV_SOURCE_AUDIT.md
docs/PHASE17_EXTERNAL_ARCHITECTURE_MATRIX.md
docs/PHASE17_MEMORY_FAILURE_MECHANISM.md
docs/PHASE17_IDENTIFIABILITY_REPORT.md
docs/PHASE17_VISUAL_JEV_TEACHER.md
docs/PHASE17_FINAL_RESEARCH_REPORT.md

以及：

reports/JEV_MASTER/MASTER_GOAL.json
reports/JEV_MASTER/EXPERIMENT_REGISTRY.json
reports/JEV_MASTER/EVIDENCE_REGISTRY.json
reports/JEV_MASTER/NEXT_ACTION.json

reports/JEV_PHASE17/EXTERNAL_CODE_MANIFEST.json
reports/JEV_PHASE17/WHO_Q4_FACTORIAL.json
reports/JEV_PHASE17/FIRST_DIVERGENCE.json
reports/JEV_PHASE17/OWNER_IDENTIFIABILITY.json
reports/JEV_PHASE17/VISUAL_JEV_COMPARISON.json
reports/JEV_PHASE17/NO_NEW_HEAD_ABLATION.json
reports/JEV_PHASE17/SOFT_MEMORY_CAUSAL_RESULTS.json
reports/JEV_PHASE17/FAIR_BASELINE_RESULTS.json
reports/JEV_PHASE17/ONLINE_TRACKING_RESULTS.json
reports/JEV_PHASE17/LATENCY_RESULTS.json
reports/JEV_PHASE17/FINAL_GO_NO_GO.json

对于未运行的实验，必须写NOT_RUN与原因。
不得填入模拟数值或伪造PASS。

最终报告回答：

Q1. Phase XVI为什么从147次IDSW退化到1494次？
Q2. WHO与Q4的记忆改变各自贡献多少退化？
Q3. hard filtering是否破坏正常身份连续性？
Q4. 当前历史证据能否识别正确Global ID owner？
Q5. Visual Jev的LM head无新增决策头方式能否提供额外身份证据？
Q6. 共享prefix和多问题执行在我们的真实MOT中有多少吞吐收益？
Q7. soft evidence weighting是否优于hard quarantine？
Q8. 普通MLP、Attention是否已经能够解释全部收益？
Q9. JEV是否取得真实、独立、可泛化的增量价值？
Q10. 三生命周期共享模型是否真的必要？
Q11. 是否具备WWW论文的可验证科学贡献？
Q12. 下一项最值得执行的实验是什么？

完成后：
运行测试。
冻结原始实验结果。
更新MASTER_GOAL.json。
提交、推送GitHub研究分支。
报告准确commit SHA和所有状态。
给出可复现命令。
明确下一阶段自动推进条件。

========================================
17. 最终指令
========================================

立即启动本次研究任务。

你的最终目的不是再次输出一份NO_GO报告，
也不是单纯把JEV改造成更大的Transformer。

最终目的是真正研究并构建：

一个能够利用可信视觉身份历史，在严格在线MCMOT中作出可靠身份关联、记忆维护和旧身份恢复决策的JEV系统。

它必须能够证明自身相对普通关联模型的可验证贡献，并形成符合WWW要求的完整科研证据。

优先解决：
Global Identity Ownership
→ Safe Identity Association
→ Reliable Identity Memory
→ Long-Term Identity Recovery
→ Unified JEV Decision Learning
→ WWW-level Evaluation

本阶段不必在每个普通工程步骤等待用户再次确认。

完成一个合格阶段后继续执行下一阶段。
如果假设失败，保存证据并进入新的有界研究假设。
如果前置科学资格失败，停止依赖它的不合格训练，而不是跳过Gate。

不要以修改门槛或重用测试数据代替真正的研究创新。

最终MASTER GOAL只有在科学结果、独立验证和论文证据全部符合预先冻结的条件时，才允许标记为SCIENTIFIC_GO或SUBMISSION_READY。

START NOW.
