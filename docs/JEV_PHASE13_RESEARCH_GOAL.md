# FINAL GOAL — WWW / JEV PHASE XIII
## JEV-MCMOT: A Native Global Identity Decision Network
### 完全替换 GMT-GTA 的视觉 Jev 多摄像头多目标跟踪研究

**Repository:** https://github.com/LYQ1107/WWW

**冻结研究基线：**

`fffd0a1b7c04a19513f0b8fa07a326572533f0ae`

**新研究分支：**

`jev/www-jev-phase13-native-gta-replacement-20261009`

## 一、最终科研目标

本次任务不是继续优化 Phase XII 的 JEV 后处理器。

我们要从根本上重新设计一个新的、真正独立承担多摄像头全局身份关联职责的 **JEV-MCMOT**。

具体目标：

**保留 GMT Stage 1 的检测和身份视觉特征学习能力，完全替换 GMT Stage 2 中的 GTA（Global Trajectory Associate）神经关联网络。**

新的 JEV 应直接接收：

- 当前摄像头的真实检测；
- 由 Stage 1 得到的视觉身份特征；
- 当前和过去所有摄像头维护的 Global Identity Memory；
- 实际时间、空间、相机和运动证据；
- 当前真实存在的合法身份候选。

直接输出：

- 当前检测属于哪个 Global ID；
- 是否需要进入历史身份恢复；
- 应恢复哪个 Stale ID；
- 是否应建立新 ID；
- 是否接受本次记忆更新（具备监督后）。

**JEV 不再依赖 GTA 先计算关联结果，也不再只是决定接受或拒绝 GMT 的匹配。**

本次必须实现、训练、验证、消融，并给出真实 HOTA、AssA、IDF1、IDSW 和在线速度结果。

不能只写设计文档，也不能只跑 Tiny。

---

## 二、明确保护的科研事实

永久保留 Phase V–XII 的所有源码、checkpoint、失败记录与结果。

Phase XII 的结论为：

- Visual JEV Full HOTA：58.524。
- GMT OFF HOTA：78.110。
- AssA：45.416 vs 80.821。
- IDSW：约 10,523 vs 204。
- 真实结构和原生执行通过。
- MATCH-only 训练成功执行，但泛化失败。
- MEMORY、REACTIVATION 尚未完成有效监督。
- Heldout video20/21/22 封存。
- Full24 和官方 TEST 未授权。

不能修改这些历史结果。

新 Phase XIII 使用独立分支、独立实验运行目录、独立数据集版本和 SHA manifest。

不要用少量被筛选出的 132 条纠错记录继续训练完整 Stage 2。

**本次主要改变的是关联模型的训练职责和训练数据分布。**

---

# M0 — 完成 GMT Stage 1 / Stage 2 代码及权重审计

必须逐函数检查：

`gtr/modeling/roi_heads/gtr_roi_heads.py`

`gtr/modeling/roi_heads/association_head.py`

`gtr/modeling/roi_heads/transformer.py`

`gtr/modeling/meta_arch/gtr_rcnn.py`

`configs/VISION_stage1.yaml`

`configs/VISION_stage2.yaml`

`configs/VISION_test.yaml`

重点审计：

1. 检测器和 DLA/BiFPN。
2. ROI feature extraction。
3. Stage 1 VFCE 身份特征训练。
4. Stage 2 RPCE 空间关系编码。
5. GTA Transformer Encoder–Decoder。
6. Association logits 和 unmatched probability。
7. 历史 proposal 到 Global ID 的轨迹聚合。
8. Hungarian。
9. Stale bank。
10. Birth、Memory、Global ID commit。

### M0.1 必须核实真实 checkpoint

检查服务器上实际存在的 Stage 1、Stage 2 checkpoint。

不能假设历史目录仍有权重。

特别注意：

`VISION_stage1.yaml` 使用 `REID: True`。

`VISION_stage2.yaml` 使用 `REID: False`。

必须核验 Stage 1 VFCE 是否真实训练完成，以及 RPCE 是否只在 Stage 2 获得有效训练。

主实验优先使用**经过真实核验的 Stage 1 身份视觉特征**。

不要直接把经过 Stage 2 训练的 1152D 表示当成未经 GTA 训练的 Stage 1 输入。

如使用 Stage 2 视觉特征，必须单列为 pretrained-transfer control。

### M0.2 冻结原始 GMT

必须保留未经修改的 GMT OFF 模式，并核对完整原始基线。

原始 GMT 的强基线与“所有新关联模型共享 Stage 1 特征”的公平比较属于两种不同实验，不得混为一谈。

输出：

`docs/JEV_PHASE13_GMT_AUDIT.md`

`reports/JEV_PHASE13/STAGE1_CHECKPOINT_AUDIT.json`

`reports/JEV_PHASE13/GMT_BASELINE_FREEZE.json`

---

# M1 — 固定 20 个开源参考项目及设计来源

根据此前审计，参考 10 个视觉 Jev 项目：

1. guanxuyu-sv/Visual-Jev
2. Liuziyu77/Valen
3. tinnel123666888/OmniJev
4. OmniJev/OneJev
5. TianyuCodings/NanoJev
6. arnodjiang/Vision-JEV
7. Xiaooolong/vev
8. IamBusy/OpenJev-Vision
9. sseanliu/Jev-Vision
10. mohit67890/imajev

再参考 10 个 MOT/MCMOT 项目：

1. TrackingLaboratory/CAMELTrack
2. MCG-NJU/MOTIP
3. MCG-NJU/MeMOTR
4. kamkyu94/TrackTrack
5. dvl-tum/SUSHI
6. NirAharon/BoT-SORT
7. FoundationVision/ByteTrack
8. GerardMaggiolino/Deep-OC-SORT
9. SysCV/qdtrack
10. noahcao/OC_SORT

为每个项目记录：

- Git commit SHA。
- 具体源码文件。
- 实际执行的关联/决策函数。
- 输入输出。
- 训练监督。
- 在线限制。
- 许可证。
- 哪些机制适合本次 GTA-free JEV。
- 哪些工作只能作为离线参考，不能直接移植。

优先参考：

**Valen：** State–Question–Candidate 分层。

**OmniJev：** 运行时选项评分与问题门控。

**MOTIP：** 直接身份解码。

**CAMELTrack：** 密集轨迹—检测关联训练。

**MeMOTR：** 长短期身份记忆。

**GMT：** 跨摄像头共享 Global ID 和原生状态执行。

不得简单复制第三方模型后更名为 JEV。

输出：

`docs/JEV_PHASE13_EXTERNAL_CODE_AUDIT.md`

`docs/JEV_PHASE13_METHOD_DIFFERENTIATION.md`

---

# M2 — 建立全新的密集身份关联训练数据集

这是本次最重要的数据任务。

以前 Phase XII 的训练集只有 132 条 TRAIN 记录，主要服务于少量关联纠错。

现在模型将负责所有身份关联，因此必须使用**完整 TRAIN 视频中的真实关联事件**进行训练。

## M2.1 冻结视频和场景分割

优先沿用此前已经冻结的 TRAIN/Validation/Heldout 协议。

- 训练数据不得包含 video17/18/19。
- 开发验证：video17/18/19。
- 独立 heldout：video20/21/22。
- 官方 TEST 不用于训练或调参。
- 不自动开启 Full24。

还要核查场景、身份和近重复帧之间的潜在数据泄漏。

开发验证视频已经被历史研究使用多次，不得重新宣称其为从未见过的独立测试集。

## M2.2 生成 Dense Global Association Dataset

从真实检测与官方训练 GT 中，构建每个时刻的：

`Current Detections → Historical Global Identities`

训练样本包含：

- 当前检测视觉特征。
- 真实历史视觉观测。
- Global Identity Bank。
- 当前相机和时间。
- 当前 bbox 与过去几何状态。
- active/stale 轨迹状态。
- 候选合法性。
- 当前 GT identity label。
- 候选身份与 GT 的关系。
- NEW/REACTIVATION 标签。
- 候选竞争关系。

必须包含大量**正常且无需纠错的日常匹配**。

同时构建困难正例、困难负例、跨摄像头关联、遮挡恢复、身份新建和错误竞争等训练子集。

不得把密集训练数据重新筛选成只有高纠错收益的样本。

GT 只能用于训练标签，不允许作为在线模型输入。

预测历史身份被多个 GT 污染时，标记 ambiguous/UNKNOWN，不允许硬编码一个错误的唯一标签。

## M2.3 候选支持审计

候选集合不能依赖 GMT GTA scores。

可以使用冻结 Stage 1 外观向量、合法时空条件或纯检索机制提供候选。

必须统计：

- Active 正确身份召回。
- Stale 正确身份召回。
- NEW 识别支持。
- 候选数分布。
- Candidate Recall@K。
- 缺失正确候选的原因。

在看到验证性能前冻结候选检索和剪枝协议。

输出：

`docs/JEV_PHASE13_DENSE_DATASET.md`

`reports/JEV_PHASE13/DENSE_DATASET_MANIFEST.json`

`reports/JEV_PHASE13/LABEL_AUDIT.json`

`reports/JEV_PHASE13/CANDIDATE_RECALL.json`

---

# M3 — 正式重新设计 JEV-MCMOT Stage 2

新目录：

`gtr/modeling/jev_stage2/`

核心网络必须具备以下结构。

## 3.1 Global Identity Memory

为每个动态 Global ID 保留：

- Short-term visual observations。
- Long-term visual prototype。
- Camera-specific appearance summaries。
- Temporal/geometry state。
- Active/Stale status。
- Historical confidence。

所有摄像头共用同一个 Global Identity Bank。

Global ID 的数值只能用于系统引用，不能作为固定训练类别输入。

## 3.2 Shared Visual State Encoder

直接使用 Stage 1 视觉特征与历史身份 Tokens。

可采用 256D 的隐空间、两层轻量 Transformer 作为初始设计。

先冻结真正的输入张量尺寸，再确定网络宽度和参数预算。

不得依赖 GMT GTA 输出。

## 3.3 Dynamic QuestionReader

真实构造：

`MATCH_ID`

`REACTIVATE_ID`

`UPDATE_MEMORY`

MATCH 的动态 Question 应包含：

- 当前检测视觉信息。
- 当前时空条件。
- 全局身份候选信息。
- 候选竞争关系。
- 跨摄像头状态。

不能只使用一个固定 Question embedding。

针对 Question 读取多个实际证据 Tokens，避免 Phase XII 中单一 Question Key 的退化注意力问题。

## 3.4 Action OptionReader

为每一个合法历史身份建立独立 Option。

每个 Option 读取：

- 当前目标视觉特征。
- 该 Global ID 的真实历史视觉。
- 动态 Question。
- 全局状态。
- 当前合法身份约束。
- 多候选竞争关系。

允许任意候选数量，并要求 permutation equivariance。

## 3.5 Typed Decision Heads

输出：

MATCH：

`P(Global ID_1, ..., Global ID_K, DEFER)`

REACTIVATION：

`P(Stale ID_1, ..., Stale ID_M, START_NEW)`

MEMORY：

`P(WRITE, KEEP)`

对于没有监督的数据类型，使用显式 UNTRAINED/FALLBACK，不允许未训练的 head 直接参与在线状态修改。

ABSTAIN/FALLBACK 与 NEW 必须严格区分。

概率校准只能在有可靠标签的作用域内声称有效。

---

# M4 — 彻底删除 GTA 的关联依赖

本阶段必须证明 JEV 真正替换了 Stage 2 的 GTA。

原始链路：

`_forward_transformer → asso_predictor → activated_association → traj_score`

新链路：

`Stage1 features → JEV State/Question/Option → identity logits → lawful assignment`

对 JEV DIRECT 模式建立明确禁用规则：

- 不允许调用旧 GTA `_forward_transformer` 生成匹配。
- 不允许使用旧 `asso_predictor` 的关联分数。
- 不允许使用 `traj_score` 作为隐含匹配先验。
- 不允许以 GTA 生成的匹配结果作为 JEV 的必需输入。

### 必须增加 GTA-bypass 测试

将 GTA 的神经前向临时替换为抛出异常的测试 mock。

在 JEV DIRECT 模式下完整执行：

- 第一帧跨摄像头初始化。
- 日常 active association。
- 未匹配检测。
- Stale-bank 查询。
- 旧身份恢复。
- 新身份出生。
- Memory 更新。
- 下一帧真实状态更新。

整个链路不能触发 GTA。

如果需要使用外观余弦、几何或运动作为 JEV 的原始证据，可以使用，但不得调用旧 GTA 的学习式评分。

原始 GMT 模式保持不变。

输出：

`reports/JEV_PHASE13/GTA_FREE_CONTRACT.json`

---

# M5 — 保留真实在线 Global Identity 执行

JEV 的概率输出不能绕过合法匹配直接修改轨迹。

保留原生状态：

- track_ids。
- id_count。
- Gallery。
- Active/Stale identity bank。
- 每个身份的历史。
- RNG。
- Camera identity scope。

必须满足：

**同一个相机、同一时刻，一个 Global ID 最多分给一个检测。**

**不同相机可以合法共享同一个 Global ID。**

保持严格在线：没有未来帧、未来标签或离线图优化。

建议第一版保留 GMT 的逐 camera payload 执行顺序，但所有视角共享身份记忆。

后续同帧多摄像头联合分配可以作为额外结构实验，不能在第一次验证时同时改变所有求解条件。

实现三种模式：

`OFF`

`SHADOW`

`JEV_DIRECT`

OFF 与历史 GMT 完全一致。

SHADOW 只运行 JEV，不修改轨迹。

JEV_DIRECT 使用 JEV 自身评分执行合法身份分配。

所有 ID、Gallery、Memory 的变化必须有状态日志和可重复回放。

---

# M6 — 第一阶段：密集监督的 JEV Stage 2 训练

不能立即进行大规模长期训练。

先执行：

`Smoke → Frozen Tiny → Dense Pilot → Formal Training`

## 6.1 Tiny Learnability

事先冻结多种真实训练场景，包括：

- 日常正常关联。
- 跨摄像头身份匹配。
- 多人相互竞争。
- Stale reactivation。
- 真正 NEW。
- Empty candidate。

要求模型能正确处理合法动态候选和可确认标签。

不得挑选简单视频来获取 Tiny PASS。

## 6.2 Dense Supervised Learning

优先使用：

`L_choice`

动态合法身份选项监督。

`L_assignment`

联合分配结构化排序监督。

`L_calibration`

有可靠监督时的概率 NLL/Brier。

第一轮不要直接混入旧 Phase XII 有限的 H32 value labels。

这些标签来自 GMT_OFF future，不一定适合新 JEV 后续策略。

建议先完成密集关联监督，验证真实在线 MATCH/NEW/REACTIVATION 能否工作，再逐步研究长期价值。

## 6.3 正式预算

采用共同的 optimizer-update 预算，而不是只比较 epochs。

以 GMT Stage 2 的原始约 20,000 updates 作为主要可比训练预算之一。

Codex 应根据实际 GPU、batch、数据规模和吞吐先执行有界 Pilot，再按冻结的正式预算训练。

至少保存三个随机种子，并且全部普通神经关联基线采用相同数据和训练预算。

不得只为 JEV 增加训练轮数。

---

# M7 — 第二阶段：On-policy Identity Learning

第一阶段如果学会正常关联，就进入真实在线历史分布训练。

让 JEV 在 TRAIN 视频上自己生成：

- Global ID。
- History Gallery。
- Active/Stale 状态。
- 真实身份选择。
- 错误出生。
- 错误合并。
- 轨迹断裂。
- 受到先前误关联污染的视觉历史。

然后使用 TRAIN GT 离线构造可靠的纠错监督。

不允许在线输入 GT。

不能将混合了多个真实身份的预测轨迹强行赋予唯一正确标签。

将正常状态与模型错误后产生的状态共同用于新的训练。

建议采用有限轮次：

`Supervised JEV → Online Train Rollouts → Audited Relabeling → Fine-tune`

这一步专门解决 Phase XII 已暴露的：

**训练在 GMT_OFF 历史下进行，但推理在 JEV 自己产生的历史下运行，导致状态分布不匹配。**

必须报告每轮训练前后的真实在线关联错误是否下降。

---

# M8 — 训练公平基线

除 GMT Original 外，必须至少建立：

**B0：GMT Original Stage 2**

原始 GTA 强基线，保持官方对应模型和训练配置。

**B1：Stage1 + Cosine/Geometry**

不使用 GTA 的轻量关联基线。

**B2：Stage1 + MOTIP-style Decoder**

历史轨迹条件化动态 ID prediction。

**B3：Stage1 + CAMEL-style Association**

检测与历史轨迹 Token 的学习式关联。

**B4：Stage1 + Set Transformer**

普通 Attention 动态候选关联。

**B5：JEV-MCMOT Full**

真正的 Shared State–Question–Option–Decision 网络。

所有方法必须：

- 使用同样的 Stage1 前端。
- 使用同样的训练视频。
- 使用同样的合法候选和 GT 监督。
- 使用同样的 native identity executor。
- 使用公平的优化预算。
- 记录实际参数量、MAC、FLOPs 和耗时。

若 JEV 使用更多的训练监督或更大的模型，必须补充 matched-capacity、matched-supervision 对照。

不能将原始 GMT 全模型结果直接当成只有关联网络变化的受控消融。

---

# M9 — JEV 结构消融

必须至少比较：

- Full JEV。
- No Dynamic QuestionReader。
- Fixed Question Embedding。
- No OptionReader。
- No Question Gating。
- No Cross-camera Identity Memory。
- No Long-term Visual Tokens。
- No Candidate Competition。
- No Typed Decision Head。
- No Probability Calibration。
- MATCH-only。
- MATCH + REACTIVATION。
- 三生命周期联合（只有相应监督合格才执行）。

同时要求：

- 所有候选排列不改变语义。
- 身份 ID 整数重命名不改变预测。
- 无未来信息。
- NEW 不等于 ABSTAIN。
- 各相机合法共享 Global ID。
- 原生状态持续一致。

特别要对比普通 Set Transformer 和 MOTIP-style：

**如果它们使用同样的视觉输入、GT 标签和训练预算就能达到同样效果，不得将增益全部归因于 JEV 结构。**

---

# M10 — 真实在线 MCMOT 评测

在前述科学 Gate 通过后，使用固定开发验证视频：

video17/18/19。

必须执行真实的：

`JEV Decision → Native Commit → Updated Memory → Next-frame Decision`

不可重放 GMT_OFF 的未来身份动作。

每个模型输出：

**Tracking：**

HOTA、AssA、IDF1、MOTA、IDSW、Frag。

**MCMOT：**

Cross-camera continuity、Global ID consistency、跨摄像头身份冲突。

官方 CVIDF1/CVMA 评测环境可用时，补充原生官方结果。

**Decision：**

Correct-ID Recall、正常关联保持率、错误身份匹配、新身份误出生、恢复错误、错误持续时间、Gallery 污染。

**Calibration：**

NLL、Brier、ECE 和风险—覆盖率曲线，注明可靠标签的评价范围。

**Efficiency：**

参数量、推理 FLOPs/MAC、Stage2 延迟、完整 FPS、p50/p95 和显存。

严格区分实时原生预测与传统基准后处理。

不要将逐视频 HOTA 简单平均替代真实 pooled TrackEval。

---

# M11 — 科学 GO / NO-GO

必须设置自动执行 Gate。

**G0 — Stage1 Feature Truth**

真实 Stage1 身份特征来源和权重有效。

**G1 — GTA-Free**

任何新 JEV 主路径均不调用 GTA 神经评分。

**G2 — Dense Supervision**

自然分布、跨视角、NEW、REACT、正确候选及视频隔离真实有效。

**G3 — Tiny Fit**

新模型真实学习合法动态身份选择。

**G4 — Native State**

真实身份提交、Gallery、Bank、RNG 与相机约束正确。

**G5 — Supervised Generalization**

在开发验证上优于简单关联基线，不发生严重身份崩溃。

**G6 — Strong Model Comparison**

与 MOTIP-style、CAMEL-style、Set Transformer 公平比较。

**G7 — Jev Independent Value**

类型化问题、候选动作、概率决策和共享身份状态提供实际可测量的独立收益。

**G8 — Online Efficiency**

实际替换 GTA 后，满足事前冻结的 Stage2/端到端实时性预算。

**G9 — Heldout Authorization**

只有结构、训练协议和开发验证全部冻结且达到科学 Gate，才允许开启 video20/21/22。

官方 TEST 和 Full24 不自动授权。

未运行的实验必须填写 NOT_RUN、reason、null metrics。

---

# M12 — 最终文件与交付

新增：

`gtr/modeling/jev_stage2/`

`reproduction_tools/build_dense_jev_stage2_dataset.py`

`reproduction_tools/train_jev_stage2.py`

`reproduction_tools/train_jev_stage2_onpolicy.py`

`reproduction_tools/evaluate_jev_stage2_online.py`

`reproduction_tools/audit_jev_stage2_gta_free.py`

`reproduction_tools/compare_jev_stage2_baselines.py`

文档：

`docs/JEV_PHASE13_RESEARCH_GOAL.md`

`docs/JEV_PHASE13_SOURCE_AUDIT.md`

`docs/JEV_PHASE13_GMT_STAGE2_CONTRACT.md`

`docs/JEV_PHASE13_DENSE_DATASET.md`

`docs/JEV_PHASE13_ARCHITECTURE_SPEC.md`

`docs/JEV_PHASE13_TRAINING_PROTOCOL.md`

`docs/JEV_PHASE13_FAIR_BASELINES.md`

`docs/JEV_PHASE13_FINAL_RESEARCH_REPORT.md`

实验目录：

`reports/JEV_PHASE13/`

至少包含：

- `FINAL_GOAL.json`
- `STAGE1_CHECKPOINT_AUDIT.json`
- `DENSE_DATASET_MANIFEST.json`
- `GTA_FREE_CONTRACT.json`
- `NATIVE_PARITY.json`
- `TINY_LEARNABILITY.json`
- `SUPERVISED_TRAINING.json`
- `ON_POLICY_TRAINING.json`
- `BASELINE_COMPARISON.json`
- `ARCHITECTURE_ABLATION.json`
- `ONLINE_VALIDATION.json`
- `EFFICIENCY.json`
- `FINAL_GO_NO_GO.json`

每项绑定：

Git commit、source SHA、dataset SHA、checkpoint SHA、配置、随机种子、运行环境及真实 evaluator 结果。

大型数据、模型权重、预测结果和状态快照留服务器。

仅推送必要源码、配置、指标、报告、图表与哈希清单。

不得删除历史 Phase V–XII 的可复现资产。

---

# FINAL EXECUTION INSTRUCTION

**现在立即开始执行 Phase XIII。**

从提交：

`fffd0a1b7c04a19513f0b8fa07a326572533f0ae`

创建独立 worktree 和研究分支。

按照：

`GMT Stage1 Audit`

→ `Dense Global Association Dataset`

→ `Architecture Freeze`

→ `True GTA-Free JEV Stage2`

→ `Native Online Identity Execution`

→ `Dense Supervised Training`

→ `On-policy State Learning`

→ `MOTIP/CAMEL/SetTransformer Fair Comparison`

→ `Real MCMOT Validation`

→ `Architecture Ablation`

→ `Final GO/NO-GO`

连续执行。

普通工程问题自行修复、测试后继续。

如果训练数据或接口存在实质性科学错误，必须 fail-closed，不得制造虚假结果。

不要只交付文档或简单 Tiny 实验。

**本次最终目标是完成一个真正由 JEV 承担 Global Identity Association 的在线 MCMOT 跟踪器，系统验证它是否能够在不依赖 GMT 原始 GTA 的情况下学会跨摄像头身份关联，并检验 Jev 风格完整决策建模相对于普通 Attention 和原始 GMT 的独立价值。**

完成后推送 GitHub，提供最终 commit SHA、每阶段 PASS/FAIL/NOT_RUN、真实模型指标、训练配置和下一步科研结论。
