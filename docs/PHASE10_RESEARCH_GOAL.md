# FINAL GOAL — WWW / JEV PHASE X
## Repair Native State Parity → Train Three Candidate Models → Fair Closed-Loop Comparison

### 0. 你的任务和执行权限

Repository：

https://github.com/LYQ1107/WWW

当前研究基线：

`jev/www-jev-phase9-native-candidate-choice-20261008`

已核验 HEAD：

`af081a9371ff0634d384d6106aa2a542e5b69afa`

建议新分支：

`jev/www-jev-phase10-native-gallery-repair-20261008`

本次任务的目标非常明确：

**首先彻底修复 Phase IX 因果 Replay 与 GMT 生产环境之间的 Gallery、Memory、Reactivation 状态差异，重新核验因果训练数据；修复验收成功后，立即开展 Candidate MLP、Candidate DeepSets、Candidate JEV 三种模型的公平训练、消融和真实闭环评测。**

这是一项连续执行任务。

不要修复完后只写一篇报告就停止，也不要重新进行已经完成的外部开源论文检索。

除非触发无法通过的科学硬门槛，否则你应自主推进后续阶段，不需要用户再次下达“继续训练”指令。

最终需要回答：

1. 真实候选条件化模型能否纠正 GMT 错误身份关联？
2. 普通 Candidate MLP 是否已经足够？
3. DeepSets 的候选间交互是否带来收益？
4. Candidate JEV 是否比 MLP 和 DeepSets 具有额外优势？
5. 当前身份正确性监督与未来 H32 因果效用监督，哪一种更有价值？
6. 关联收益到底来自更好的候选表示、训练监督、全局求解器，还是 JEV 特定结构？

禁止为了得到正向结果放宽之前的科学门槛。

---

# 1. 保留 Phase V–IX 证据

永久保护：

Phase V B2 checkpoint SHA256：

`f2aa3dd2b564d90bfbfb62dc0518b0b2107a931ed7134f8c0d19f5d152b94ed7`

Phase VII：

现有 state-only JEV 没有在公平比较中证明优于普通固定规则。

Phase VIII：

原始数据 Gate FAIL，不能修改历史结论。

Phase IX：

`STOP_BLOCKED_DEPLOYMENT_CONTRACT`

已完成：

- Train verified corrections：74。
- Validation verified corrections：59。
- Train independent conflict groups：17。
- Validation independent conflict groups：9。
- 生产候选接口单元测试：20 PASS。
- video12 GMT OFF 完整兼容：1200 帧 PASS。
- 候选值实际改变 native ID commit：PASS（有界回归）。
- 后续在线关联分数受到状态变化影响：PASS（有界回归）。
- 221 个 Gallery 状态快照中，birth-only 修复仍有 94 个不一致。
- Candidate MLP / DeepSets / JEV 未训练。
- Heldout video20/21/22 未使用。
- 官方 TEST 未使用。

不得覆盖、删除、重新命名已有冻结 checkpoint、原始标签或实验结果。

先确认服务器 worktree：

`/home/liuyeqiang/WWW_jev_phase9`

以及 runtime：

`/home/liuyeqiang/WWW_jev_phase9_runtime/20261008_v1`

实际存在。

如果目录变化，首先通过 Git/manifest 找到正确路径，不能凭推测使用其他数据。

建立新的独立 worktree 与运行目录。所有新实验绑定固定 source commit、config、seed、cache SHA 和 checkpoint SHA。

---

# 2. PHASE A — 彻底修复 Native Gallery / Replay 状态

这是本次最高优先级。

## A1. 重新阅读错误来源

必须逐行检查：

- `gtr/modeling/meta_arch/gtr_rcnn.py`
- `gtr/modeling/jev_candidate_features.py`
- `gtr/modeling/jev_candidate_assignment.py`
- `gtr/modeling/jev_candidate_policy.py`
- `reproduction_tools/jev_counterfactual_v2.py`
- `reproduction_tools/run_early_pilot_tracking.py`
- `reproduction_tools/jev_phase6_rollouts.py`
- `reproduction_tools/jev_phase8_candidate_submit.py`
- `reproduction_tools/jev_phase9_gallery_bridge.py`
- `reproduction_tools/check_jev_phase9_feature_bridge.py`
- `reproduction_tools/check_jev_phase9_gallery_restoration.py`
- `reproduction_tools/check_jev_phase9_memory_representation.py`
- `reproduction_tools/audit_jev_phase9_gallery_failure.py`

重点明确 GMT production 中：

1. 新身份创建时，第一条 ReID 向量如何进入 Gallery。
2. 已存在身份匹配成功后，何时写入 Gallery。
3. 哪些情况下 MEMORY 决定 WRITE/SKIP。
4. Stale-bank REACTIVATION 发生于哪个阶段。
5. 旧身份恢复后的观测是否写入 Gallery。
6. 每条身份的 hits、memory count、history length 如何变化。
7. Gallery 的平均、归一化和实际检索逻辑。
8. 生产的状态提交与 Replay 的写入顺序是否一致。

必须特别复核已知事件：

`video12, frame839, view1, identity5`

以及：

`video12, frame851, view1, identity5`

原生系统存在从 stale bank 恢复旧身份并写入观测的操作，而此前 Replay 漏掉了部分写入。

已知统计：

- 39,296 个 track-prefix instances 差 1 条观测。
- 234 个差 2 条。
- 6 个差 3 条。
- 共 39,536 个 instances。
- 221 个 snapshot，birth-only 修复失败 94 个。

这些是当前需要解释和修复的已知差异，不允许把它们直接视为正常浮点误差。

## A2. 修复原则

**优先使用真正的 Native Production State Snapshot 作为因果分支起始状态。**

首先评估能否从生产运行中直接保存必要的可变状态：

- 完整 ordered Gallery / ReID 向量。
- ID 到历史身份的运行时映射。
- Track history、hits 和 age。
- 当前 association history。
- Active / stale IDs。
- Memory bank。
- 各身份的实际首次观测和后续写入。
- 已提交的当前身份。
- Birth / reactivation 状态。
- 当前视角与帧状态。
- RNG state。
- 影响未来关联的其他 mutable metadata。

需要明确原生状态是在当前 MATCH 之前、之后，还是完整提交之后采样，不得混淆。

推荐实现一个可版本化的：

`NativeProductionPrefixRecorder`

配合：

`NativeStateForkAdapter`

允许从相同 production prefix 出发，执行多个合法干预，并在后续 H32 中使用真实 online mutable state。

如果直接保存生产状态不可行，可以修复 Replay adapter，但必须实现完整的事件语义，而不是只补一个首次向量。

建议将身份观测事件明确区分为：

- BIRTH_INITIALIZE
- EXISTING_MATCH_WRITE
- MEMORY_WRITE
- MEMORY_SKIP
- BANK_REACTIVATE
- REACTIVATION_WRITE
- TRACK_LOST / STALE
- NEW_ID_COMMIT

事件日志记录真实来源帧、相机、候选行及真实 ReID 向量引用。

任何补充向量必须来自冻结真实 perception cache，不允许捏造或用零向量填补。

不能将 `track_hits` 直接当成 Gallery 长度，也不能假设所有身份永远满足 `gallery_length = hits`。

对于需要聚合的 Gallery，必须使用生产实际规定的向量集合和顺序。

## A3. 原生状态完整等价测试

至少复核 Phase IX 已有全部 221 个固定快照。

不得更换事件集合。

重点覆盖原 94 个失败快照，以及 birth、普通关联、stale 恢复和多次恢复场景。

同时检查模型输入中的 64D online state 和 12D candidate evidence。

验证：

- Video/frame/view/event key 一致。
- 当前检测和候选 ID 顺序一致。
- GMT raw association scores 一致。
- 原始 Hungarian proposal 一致。
- 所有 candidate legal masks 一致。
- Current state64 一致。
- Candidate evidence12 一致。
- Gallery 长度一致。
- Gallery 实际向量内容和顺序一致。
- Gallery cosine 一致。
- Track hits 与 age 一致。
- Bank eligibility 一致。
- Stale / active state 一致。
- 原生 committed IDs 一致。
- 后续 Memory/Birth 状态一致。
- RNG state 一致。

离散字段、候选 ID、合法 mask、状态事件序列要求精确一致。

浮点输入在相同运算定义下要求：

`max_abs_error <= 1e-6`

若由于硬件浮点执行顺序无法逐位相同，应记录明确的数值容差和来源，不能把语义错误归入容差。

**硬门槛：所有固定快照必须通过，不能只通过 video16 的某一个成功前缀。**

输出：

`PHASE10_NATIVE_STATE_CONTRACT.md`

`PHASE10_GALLERY_PARITY.json`

`PHASE10_FEATURE64_PARITY.json`

`PHASE10_EVIDENCE12_PARITY.json`

`PHASE10_STATE_EVENT_LEDGER_AUDIT.json`

---

# 3. PHASE B — 重新验证 H32 因果数据

Gallery 修复以后，必须判断原来保存的反事实标签是否仍然有效。

不能默认：

“输入特征修正了，所以以前所有 utility 标签也正确。”

## B1. 保留旧样本选择

训练视频：

12、13、14、16。

验证视频：

17、18、19。

独立 heldout：

20、21、22。

原 Phase VIII 和 Phase IX 预选事件集合、冲突组与权重全部冻结。

不要重新按最终 HOTA 或 H32 结果选视频。

## B2. 重新执行 Native Counterfactual Forks

每个已有可核验事件从相同真实生产状态出发，按原始预注册的合法动作执行：

- CONTROL
- KEEP / ACCEPT
- REASSOCIATE
- START_NEW
- REAL_CORRECT_CANDIDATE
- 已有的其他合法候选
- 合法时的 conflict-group joint assignment

只执行真正合法的候选，不得为满足标签数量增加不存在的身份。

每个分支执行：

H8 / H16 / H32。

之后的全部检测、匹配、记忆写入、旧身份恢复都必须依照相同固定后续策略在当前被干预的状态下重新计算。

禁止复用冻结 OFF 的未来动作序列。

GT 只能用于离线标签和效用评价，不得进入模型推理输入。

## B3. 因果一致性 Gate

必须证明：

- CONTROL 对应原生事实路径。
- 显式 factual KEEP 与原生事实状态一致。
- 对同一分支，生产语义和修复后 Replay 的 committed IDs 一致。
- Gallery / Bank / Birth / Stale state 一致。
- 同一外生 perception 和随机性契约得到一致的后续结果。
- H32 未来窗口和 camera/frame 对齐正确。
- 不把 unknown GT 当成正确恢复。
- 不把未执行候选的 Q 值填零。
- 不将 overlapped causal windows 当成独立样本。

如果修复后某些 H32 结果变化：

保存 old vs new 对照。

新建 dataset version，并明确标记原标签不再满足当前 native contract。

不得覆盖原报告或隐瞒差异。

如果发现原训练/验证纠错样本数量发生变化，需要按冻结规则重新判断 Gate，不能继续沿用未经复核的 74/59。

## B4. 正式训练数据冻结

构建：

`Native Candidate Association Dataset v3`

数据包含：

- 完整合法候选 evidence。
- Current state64。
- Per-candidate evidence12。
- Candidate mask。
- 真实 NEW option。
- Candidate correctness label。
- 执行过的候选对应 H32 causal utility。
- Unknown / tie / censor masks。
- Conflict-group ID。
- 原始 GMT proposal。
- 采样来源与权重。
- Provenance / snapshot / branch SHA。

分别保留：

1. Natural distribution。
2. Hard corrective cases。
3. Hard negative / regression cases。
4. Unknown / censored cases。

注意 Phase IX 中已知 NEW correctness-positive 标签不足，不能人为创造正样本。

因此正式三模型的主实验优先固定相同的 NEW/birth 规则，仅比较已有身份候选的学习式选择。

完整 learnable NEW 可以作为额外条件实验，但只有真实正负监督足够时才允许训练并主张出生决策泛化。

不得为了训练网络，把不确定的新身份标签强行转换为正确或错误。

数据冻结后输出：

`NATIVE_CANDIDATE_V3_MANIFEST.json`

`NATIVE_CANDIDATE_V3_AUDIT.json`

`PHASE10_H32_REPLAY_PARITY.json`

`PHASE10_DATA_ELIGIBILITY.json`

数据与状态两道 Gate 完整通过以后，**立即自动进入 Phase C，不要停留在审计报告。**

---

# 4. PHASE C — 三种候选模型的 Tiny-Set Overfit

这一步用于验证模型和数据链路，不是最终性能实验。

首先使用固定的少量真实、已核验的训练事件组。

比较：

### Model A — Candidate MLP

输入：

`state64 + candidate evidence12 + shared candidate context`

采用共享的 per-candidate scorer。

必须处理动态候选个数和候选 mask。

MLP 必须能够获取与 JEV 相同的合法上下文，不能只给一个最简单的分数，以免产生不公平对照。

### Model B — Candidate DeepSets

使用共享候选编码器和 masked set pooling。

每个候选同时获得其自身证据和集合上下文。

必须支持：

- 可变候选数。
- 候选排列等变性。
- Empty / all-masked。
- Candidate rank 变化。
- 跨候选竞争。
- 合法 NEW。

### Model C — Candidate JEV

保留 Question/Action-conditioned 结构。

对于 MATCH，候选不是固定身份分类编号，而是当前合法动态候选。

建议使用：

- Shared online state encoder。
- Candidate evidence encoder。
- Candidate-context aggregation。
- Semantic candidate/action scoring。
- Legal-action masking。
- Candidate correctness head。
- Optional causal utility head。

初始规模以现有 B2 约 34K 参数为参考，但不要求机械固定；记录实际参数和计算量，并构建对应的 capacity-matched MLP 与 DeepSets。

三种模型都必须获得相同输入、候选和合法动作。

**不得因为 JEV 有双头而让 MLP/DeepSets 只能输出单头。**

先检查：

- Tiny training loss 能否下降。
- 正确候选是否能够被记住。
- 未见候选位置排列时是否保持一致。
- Mask 是否正确。
- Float finite。
- 训练和生产归一化一致。
- Candidate value 是否能进入同一个原生求解器。

若所有模型都无法拟合：

先排查标签、candidate identity anchors、输入归一化、mask、数据混叠。

若只有某一种失败：

先核对该模型的优化与输入契约，再判断架构表示是否不足。

不要直接增加 epochs 来掩盖错误。

通过后自动进入 Phase D。

---

# 5. PHASE D — 三模型正式公平训练

本阶段必须真实训练以下三种模型，而不是只生成配置：

1. Candidate MLP
2. Candidate DeepSets
3. Candidate JEV

使用相同：

- Train split。
- Validation split。
- Native dataset v3。
- Candidate evidence。
- Legal action mask。
- NEW 规则。
- Global assignment solver。
- Native state commit。
- Feature normalization。
- Optimizer/update budget。
- Training seed protocol。
- Calibration protocol。
- Model selection metric。

推荐以 Phase VIII 的预注册学习预算作为起点：

- AdamW。
- LR 0.001（除非已有预先固定的修订协议）。
- Weight decay 1e-4。
- Batch 32。
- Gradient clip 5。
- Seed 20261008、20261009、20261010。

训练单位按照独立 conflict groups 组织，防止同一段连续身份错误反复出现造成虚假的样本量。

初始共同训练预算可使用 20 epochs，并保存 best/last。

如果明显欠拟合，允许按事先冻结的相同规则扩大到 50/100 epochs，但不得只给 JEV 更多优化机会。

每个模型至少输出：

- 实际参数量。
- FLOPs/MACs。
- Training/validation loss。
- Candidate Rank-1。
- Candidate MRR。
- Corrective recall。
- Wrong-correction rate。
- Natural distribution metrics。
- Hard-case metrics。
- Runtime action frequency。
- Calibration NLL/Brier/ECE。
- 模型与数据 SHA。

## D1 — 正确性监督对比

在相同数据和结构下比较：

Candidate Correctness CE。

目标是判断模型是否能识别正确历史身份。

## D2 — 长期效用监督对比

在相同数据和结构下比较：

H32 Causal Utility / Advantage Ranking。

必须处理候选未执行、效用未知或被 censor 的情况。

不可将未执行候选作为负样本。

## D3 — Multi-task 对比

Correctness + Causal Utility。

所有三种模型均采用相同监督组合和任务损失权重。

不能只让 JEV 使用未来信息训练，再将普通 MLP 只用 correctness 训练。

## D4 — JEV 特定结构消融

至少比较：

- Same-evidence Candidate MLP。
- Same-evidence DeepSets。
- Candidate JEV。
- Candidate JEV without Question/Action embeddings。
- Candidate JEV without candidate-context interaction。
- Candidate JEV without causal utility。

注意只有 MATCH 一个固定 WHO 问题时，Question embedding 可能等价于一个常量参数。

不能在缺少多 Question 实验的情况下宣称已证明问题条件化带来跨任务迁移。

如果移除 Question/Action embeddings 后效果基本不变，需要在论文中如实说明，不得强行宣称其为必要组件。

## D5 — Candidate Top-K

之前真实纠错候选包含较深排名。

不能默认 Top-8 足够。

先在 TRAIN 统计：

Recall@8 / 16 / 32 / 64 / Full。

根据候选召回和在线时间成本冻结 Top-K 规则。

对于被截断而丢失正确身份的样本，必须单独统计为 Candidate Miss，不能归咎于 JEV 选择器错误。

---

# 6. PHASE E — 同求解器公平比较

三模型训练完成后，使用同一个：

`assign_candidate_values`

及同一个实际原生状态提交接口。

不要让 Candidate JEV 使用专有 Hungarian，而普通 MLP 使用另一个弱化求解器。

特别区分：

- GMT OFF：原始矩形 Hungarian 与阈值语义。
- Fixed rule：普通固定候选规则。
- Dynamic threshold：状态条件化规则。
- Bidirectional normalization：便宜的候选竞争对照。
- Candidate MLP。
- Candidate DeepSets。
- Candidate JEV。

GMT OFF 可作为历史跟踪参照，但它的求解目标与增强 values solver 不同，因此不能只依据 GMT OFF 与 JEV 的差值声称架构贡献。

真正的学习式架构归因必须在相同增强求解器条件下进行。

所有方法的：

- Candidate ID support
- Evidence mask
- New option
- Association constraints
- Memory/Birth semantics
- Inference state

都必须一致。

先使用 Validation 选择模型。

不得提前打开 video20/21/22。

---

# 7. PHASE F — 真实闭环 Heldout

只有以下条件全部满足：

- Native state parity PASS。
- H32 causal replay parity PASS。
- Data eligibility PASS。
- Tiny-set fitting PASS。
- 三模型训练及校准完成。
- 模型配置与 checkpoint 冻结。
- 全部比较规则预注册。

才可开启：

- video20
- video21
- video22

这些视频不参与结构设计、checkpoint 选择或参数调优。

执行各模型的完整 mutated-state online tracking。

所有后续帧都必须从当前真实状态计算新关联。

禁止套用 OFF 的未来轨迹动作。

报告：

| 指标类别 | 必须输出 |
|---|---|
| Tracking | HOTA、AssA、IDF1、MOTA、IDSW、Frag |
| Candidate | Rank-1、MRR、Recall、候选缺失率 |
| Correction | N01/N10、Corrected / Broken matches |
| Causal | H32 utility、regret、wrong-ID duration |
| Lifecycle | False birth、wrong write、bank contamination |
| Efficiency | 参数量、MAC、p50/p95 latency、显存 |

报告逐视频结果及真实 pooled TrackEval。

不要把各个视频的 HOTA 简单算术平均作为官方 pooled 指标。

参考既有预注册架构 Gate：

- 相对最强普通模型 pooled HOTA 至少 +0.1。
- pooled AssA 至少 +0.2。
- 至少两个 heldout 视频主指标正向。
- 纠错收益没有被反改错抵消。
- p95 新增关联延迟不超过 10 ms。
- 额外显存不超过 128 MiB。

如有多 seed，报告均值、方差及视频/场景级不确定性。

不能将重叠事件当作独立重复计算显著性。

如果 Candidate MLP 或 DeepSets 最好，明确报告，不能为了保留 JEV 故事改变协议。

---

# 8. 如何根据结果决定未来三模块 JEV

最终科研目标仍然是：

MATCH — WHO?

MEMORY — WRITE?

REACTIVATION — OLD OR NEW?

但这次先要求 MATCH 得到独立科学证据。

判断：

### 情况 A：JEV 优于 MLP 和 DeepSets

研究 JEV 的候选动作表示与长期效用机制，分析具体增益来源。

如果机制成立，再将共享核心扩展到 MEMORY/REACTIVATION。

### 情况 B：DeepSets 或 MLP 与 JEV 接近

说明 Candidate Conditioning 本身可能解释主要收益。

保留有效候选关联结果，但不能声称 JEV 结构必需。

应重新审视长程决策训练、共享生命周期信息和真正的状态干预是否能提供独立价值。

### 情况 C：所有候选网络效果都不好

检查：

- Correction/Regression balance。
- Natural distribution shift。
- GT anchor ambiguity。
- Candidate miss。
- Model values 与 assignment objective。
- Native state distribution drift。
- H32 utility 与真实 HOTA 是否一致。

不立即加大网络容量，也不自动扩大数据到 Full24。

---

# 9. 执行与故障恢复要求

这次不要重复“补一个点、做一个报告、然后停止”的研究循环。

对于普通工程问题：

- 自动定位源码。
- 运行最小复现。
- 修改修复。
- 编写回归测试。
- 保存旧错误报告。
- 重新跑相关受影响测试。
- 成功后进入下一阶段。

如果 birth-only adapter 不足以修复 Gallery，主动采用 Native Production Prefix Snapshot / Fork 的替代技术路径，而不是继续堆积不能证明无损的特殊补丁。

对于不可通过的科研硬门槛：

明确 BLOCKED 与原因，不准使用不可信数据继续训练或打开 heldout。

不要以节省 GPU 时间为由提前跳过状态验证。

GPU 优先使用空闲设备，不中断无关用户进程。

可以并行做相互独立的 CPU 审计、代码编写和不接触 heldout 的回归测试。

长任务必须具有：

- Resume
- Atomic completion manifest
- Exact source/checkpoint SHA
- Partial log preservation
- Deterministic replay
- Fail-closed gate
- No duplicate work

不要在同一开发 worktree 上边修改生产逻辑边运行长期 canonical 实验。

冻结用于实验的代码 commit，确保所有工作进程使用相同源码。

---

# 10. 最终交付

必须创建：

`docs/PHASE10_NATIVE_STATE_CONTRACT.md`

`docs/PHASE10_GALLERY_REPAIR_REPORT.md`

`docs/PHASE10_CAUSAL_DATASET_V3.md`

`docs/PHASE10_THREE_MODEL_EXPERIMENT.md`

`docs/PHASE10_ARCHITECTURE_COMPARISON.md`

`docs/PHASE10_FINAL_RESEARCH_REPORT.md`

以及：

`reports/JEV_PHASE10/GALLERY_PARITY.json`

`reports/JEV_PHASE10/NATIVE_STATE_PARITY.json`

`reports/JEV_PHASE10/H32_CAUSAL_PARITY.json`

`reports/JEV_PHASE10/DATA_ELIGIBILITY.json`

`reports/JEV_PHASE10/TINY_OVERFIT.json`

`reports/JEV_PHASE10/MLP_RESULTS.json`

`reports/JEV_PHASE10/DEEPSETS_RESULTS.json`

`reports/JEV_PHASE10/JEV_RESULTS.json`

`reports/JEV_PHASE10/SUPERVISION_ABLATION.json`

`reports/JEV_PHASE10/ARCHITECTURE_ABLATION.json`

`reports/JEV_PHASE10/HELDOUT_RESULTS.json`

`reports/JEV_PHASE10/FINAL_GO_NO_GO.json`

对于因科学 Gate 未通过而不能执行的阶段，明确写 NOT_RUN 和原因，禁止填充虚假的结果。

对于实际完成的模型实验，必须提交对应 checkpoint hash、训练配置、可复现命令和完整结果来源。

GitHub 只推送必要代码、配置、compact metrics、测试、文档和 SHA manifests。大型 checkpoint、state snapshots、完整 fork records 留在独立服务器目录。

完成后提交、推送研究分支，并给出准确的 commit SHA 与最终状态。

---

# 11. 最终必须回答的七个问题

1. Phase IX Gallery 差异是否被彻底修复？221 个原始固定前缀现在是否全部通过？
2. 修复以后，哪些 H32 因果标签发生了变化？数量、原因及影响是什么？
3. Candidate MLP 是否能够学会真实身份纠错？
4. DeepSets 是否显著优于 Candidate MLP？
5. Candidate JEV 是否在相同训练预算下优于两种普通模型？
6. Correctness、Causal Utility、Multi-task 中哪一种监督最有效？
7. 是否已有足够证据将 MATCH 扩展为完整的 MATCH–MEMORY–REACTIVATION Unified JEV？

结论必须区分：

- 科学假设被支持。
- 工程实现已通过。
- 数据仍不足。
- 实验尚未执行。
- 研究假设被证伪或暂不支持。

不要把代码测试通过当成模型成功。

不要把单一训练视频提升当成泛化证明。

不要把候选级模型的收益自动归因于 JEV 独特结构。

---

# 最终执行指令

**立即从 Phase IX 最新冻结结果启动 Phase X。**

执行顺序：

`Native Gallery Repair`
→ `221/221 Prefix Parity`
→ `H32 Native Counterfactual Replay`
→ `Dataset v3 Freeze`
→ `Candidate MLP Tiny Overfit`
→ `DeepSets Tiny Overfit`
→ `Candidate JEV Tiny Overfit`
→ `Three-Model Fair Training`
→ `Supervision & Architecture Ablations`
→ `Sealed Heldout Closed-Loop`
→ `Final GO/NO-GO`

通过一阶段立即进入下一阶段，不要仅因为完成了文档或 Git 提交就结束。

如果受阻，优先穷尽有明确边界、可验证的最小修复方案；不能用未经验证的近似替代真实生产状态。

**最终目标不是再完成一次数据审计，而是真正得到三种候选关联模型在相同 GMT 在线系统中的可信对比结果，并明确判断 JEV 是否具有独立的研究价值。**
