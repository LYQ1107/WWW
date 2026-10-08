# FINAL GOAL — WWW / JEV PHASE IX
## Native Candidate Choice & Causal Association Learning
### 真实身份候选选择、因果监督与结构化 JEV 的公平验证

Repository: https://github.com/LYQ1107/WWW

目标会议：The Web Conference（WWW）

研究任务：Online Multi-Camera Multi-Target Tracking

基础跟踪器：GMT / VisionTrack

上一阶段：
`jev/www-jev-phase8-corrective-association-20261008`

核验过的 Phase VIII HEAD：
`4190b803ed83bde1a44a015d15d03ceff68e03d0`

新研究分支建议：
`jev/www-jev-phase9-native-candidate-choice-20261008`

---

# 0. 最终科研目标

你需要独立完成一次真实的、可复现的研究验证。

**核心问题：在相同 GMT 感知输入、相同合法候选、相同原生全局关联与状态提交机制下，Candidate-Conditioned JEV 是否能够比固定规则、动态可学习阈值、普通 MLP 和候选集合网络做出更好的长期身份关联决策？**

我们不希望只学习：

“当前匹配要不要接受？”

最终希望学习：

“当前检测与哪些已有身份具有可信的关联关系？在当前候选竞争和未来状态影响下，应该选择哪个合法身份，或者创建新身份？”

研究需要区分：

1. 候选关联评分本身的收益。
2. 全局 assignment / Hungarian 的收益。
3. 监督目标和样本组织的收益。
4. JEV 特定结构的额外收益。
5. 后续 MEMORY / REACTIVATION 生命周期依赖的潜在收益。

不能因为把一个普通 MLP 换了名称，就声称实现了 JEV 特有的推理机制。

不能因为实现了 Candidate Attention，就宣称已经超过 MOTIP、CAMELTrack 或其他可学习关联模型。

最终的研究结论必须基于公平受控实验。

---

# 1. 严格保留已有科研证据

永久保护以下结果：

Phase V B2 checkpoint SHA256：

`f2aa3dd2b564d90bfbfb62dc0518b0b2107a931ed7134f8c0d19f5d152b94ed7`

Phase VII：原 state-only MATCH 没有证明优于普通固定规则。

Phase VIII：

- TRAIN12/13/14/16：122211 个 MATCH events。
- Validation17/18/19：38887 个 MATCH events。
- 初筛有可行正确候选：Train 2278，Validation 118。
- 原生因果审计：Train 82、Validation 13。
- 核验纠错：Train 74、Validation 13。
- 冻结数量门槛：Train >=50，Validation >=20。
- Validation gate FAIL。
- 新 candidate model NOT TRAINED。
- 完整 deployed candidate-values interface NOT VERIFIED。
- Heldout20/21/22 SEALED。
- Full24 NOT AUTHORIZED。
- Official TEST NOT USED。

必须完整保留原始结果，不允许事后把 Phase VIII 的 BLOCKED 改为 PASS。

使用新的独立 worktree 和输出目录。

不得覆盖 Phase V/VI/VII/VIII 的 checkpoint、原始预测、实验标签、manifest 和报告。

不得为了释放空间删除任何不能证明完全无科研依赖的 checkpoint。

此前存储审计检查了 656 个 checkpoint/state 文件，约 37.8 GiB，没有证明存在安全孤儿文件，因此执行删除数为零。本阶段默认继续保护。

---

# 2. PHASE 0：逐仓库学习相关开源实现

正式改代码之前，必须对照现有 WWW 项目审计以下开源代码。

所有外部仓库记录实际固定 commit、许可证、阅读路径、关键函数和适用边界。

优先阅读，不急于复制和安装其他项目的大型训练环境。

## 2.1 CAMELTrack

https://github.com/TrackingLaboratory/CAMELTrack

参考审阅 commit：
`46a74bb22a28d2d699b4c5c5e317a26d3b87f1e2`

重点：

- `cameltrack/architecture/gaffe.py`
- `cameltrack/architecture/temporal_encoder.py`
- `cameltrack/camel.py`
- `cameltrack/train/dataset.py`
- `cameltrack/train/sampler.py`
- `cameltrack/utils/assignment_strats.py`

研究其：

- Tracklet / detection token 表示。
- Group-aware feature interaction。
- Padding masks。
- OcclusionSampler、GapSampler。
- Association-centric training。
- Hungarian 与模型评分分离。

需要回答：

为什么 CAMELTrack 能利用上下文学习关联？我们的方法与其有何真正不同？

尤其学习它如何构建困难训练样本。

注意它的原始训练采样方式和我们从 native rollout 得到的反事实效用监督不是一回事，不要直接把开源采样器复制到 VisionTrack。

## 2.2 MOTIP

https://github.com/MCG-NJU/MOTIP

参考 commit：
`ffc0e905ac196a603027eca8d18fb0dff48c8bcc`

重点：

- `models/motip/id_decoder.py`
- `models/motip/trajectory_modeling.py`
- `models/runtime_tracker.py`

阅读 ID context、history embedding、cross-attention、temporal mask、身份解码与新身份处理。

需要回答：

JEV 如果直接对动态候选身份打分，与 MOTIP 的 in-context ID prediction 有什么区别？

不能把绝对 track ID 当固定分类标签来记忆。

## 2.3 TrackTrack

https://github.com/kamkyu94/TrackTrack

参考 commit：
`ee7f1c5fcbdcac48ed8bfab38d52c0006bf304da`

重点：

- `3. Tracker/trackers/tracker.py`
- `3. Tracker/trackers/utils.py`

研究：

- Track-perspective association。
- Iterative assignment。
- Track-aware initialization。
- False birth suppression。

将其作为 START_NEW 和候选竞争的规则参考。

## 2.4 BoT-SORT / Deep OC-SORT / OC-SORT

https://github.com/NirAharon/BoT-SORT

https://github.com/GerardMaggiolino/Deep-OC-SORT

https://github.com/noahcao/OC_SORT

重点：

- `tracker/bot_sort.py`
- `tracker/matching.py`
- `trackers/integrated_ocsort_embedding/ocsort.py`
- `trackers/ocsort_tracker/association.py`

学习：

- Appearance/motion fusion。
- Multi-stage assignment。
- Detection-confidence weighting。
- EMA appearance update。
- Track refinding。
- Observation-centric recovery。

区分普通工程关联改进与 JEV 学习决策的额外价值。

## 2.5 SUSHI

https://github.com/dvl-tum/SUSHI

重点：

- `src/models/mpntrack.py`
- `src/tracker/hicl_tracker.py`

研究候选图和冲突关联如何共同建模，尤其是边之间的相互影响。

禁止将使用未来视频片段的离线图推理直接移植到当前在线 MOT 中。

## 2.6 MeMOTR

https://github.com/MCG-NJU/MeMOTR

重点：

- `models/query_updater.py`
- `models/memotr.py`

研究：

- 置信度控制的轨迹表征更新。
- 短期与长期记忆融合。
- Memory attention。
- 轨迹状态更新机制。

它主要服务于后续 MEMORY 设计，不应干扰本阶段 MATCH 的受控比较。

## 2.7 QDTrack

https://github.com/SysCV/qdtrack

重点：

`qdtrack/models/trackers/quasi_dense_embed_tracker.py`

研究：

- `match()` 中的双向 softmax。
- 候选竞争。
- `update_memo()` 中的 momentum 更新。
- 简单匹配分数如何取得较好身份关联效果。

增加一个可选的 bidirectional-normalized score 控制，检查普通候选竞争归一化是否足够解释 JEV 的收益。

## 2.8 StrongSORT / Set Transformer

https://github.com/dyhBUPT/StrongSORT

https://github.com/juho-lee/set_transformer

重点：

- StrongSORT 的 `deep_sort/tracker.py`。
- Set Transformer 的 `modules.py`。

研究 matching cascade 和可变集合表示。

必须自行加入动态候选 mask、空集合保护、合法 NEW、候选排列等变性测试。

不要把没有未来约束的 post-hoc tracklet linking 当成在线方法。

## 2.9 开源审计交付

生成：

- `docs/PHASE9_EXTERNAL_CODE_AUDIT.md`
- `docs/PHASE9_METHOD_DIFFERENTIATION.md`
- `reports/JEV_PHASE9/EXTERNAL_REPOSITORY_MANIFEST.json`

必须包含代码路径和函数级分析，不允许只复制 README 或论文摘要。

---

# 3. PHASE A：验证集补充纠错数据

这是本阶段最重要的数据工作。

不要更换 Phase VIII 的训练/验证分割。

训练：
video12、13、14、16。

验证：
video17、18、19。

独立 controller-heldout：
video20、21、22。

后者在正式模型选择与训练资格确认以前必须保持 sealed。

## A1. 保留旧 Gate，不追认通过

原 Validation 13/13 不变。

生成新的 supplemental protocol，例如：

`PHASE9_SUPPLEMENTAL_CAPTURE_PROTOCOL.json`

该协议必须在读取新候选事件的未来因果结果之前冻结。

不允许：

- 根据 H32 效用选择有利事件。
- 根据模型训练结果重选验证视频。
- 删除原来不利或不确定的样本。
- 将相邻重复错误当成独立训练机会。
- 用 GT 产生运行时身份候选。

新协议只能利用已存在的观测性数据、合法候选和事先确定的分组规则进行选择。

## A2. 分组抽样

复用 Phase VIII 的 118 个验证候选可行事件索引。

按以下原则形成冲突组：

- 相同 video。
- 相同历史目标身份。
- H32 内时间相关事件。
- 共享候选轨迹的冲突关联。
- 同一状态下的多个 detection。

必须保留 group ID。

在不同 group 和视频之间分配补充样本，不得只集中抽取最容易成功的某个视频。

建议先冻结一个有界的 36–48 个补充候选事件预算；这是审计预算，不是预计会产生的有效正例数。具体上限可以根据事先可计算的分组数量与可用状态决定，但不能用未来 utility 反向决定。

所有被选事件，包括失败、unknown、tie、wrong commit，都必须保留。

必须分别报告：

- original13
- supplemental events
- combined raw count
- effective sample size
- independent conflict groups
- per-video support
- true native corrective count
- H32 positive effect
- zero-birth-penalty positive effect

只有按照新协议明确达成验证纠错支持要求，才能声称新的 supplemental gate PASS。

原 Phase VIII FAIL 永久保留，不得更改。

## A3. 补充 Natural 和 Hard Negative

不允许只收集原 GMT 错、替代身份对的事件。

同时生成：

Natural Distribution：

原生 GMT 全事件分布的预注册样本。

Hard Negative：

GMT 原本正确，但替代候选分数接近、容易错误重新关联的事件。

Hard Corrective：

GMT 原本错误，存在真实合法正确候选，且执行后有实际收益。

Unknown/Censored：

无法确定历史身份的事件，不强制指定正负标签。

所有样本必须明确标记分布来源和采样概率。

训练可以分层采样，但自然分布的验证指标必须独立报告。

不要人为制造正例或重标不确定身份。

## A4. 数据监督分为两类

Candidate Correctness：

候选是否与正确历史身份对应。

Future Utility：

执行某合法动作并经过真实原生状态提交后，未来 H8/H16/H32 的因果收益。

两者必须保留独立标签。

未来身份效用只能在真实执行过的干预分支上观测。未执行候选的效用必须为 UNKNOWN，不得填零或假装不如最优分支。

存在 multiple correct ID aliases 时，必须依据冻结 identity anchor 规则保留歧义。

不要使用 GT 的整数值直接比较预测 track ID。

## A5. 通过条件

完整要求：

- 原生候选存在且合法。
- 原生当前身份成功提交。
- 决策确实改善当前身份。
- H32 utility 相对 CONTROL 提高。
- 移除 false-birth penalty 后仍为正。
- Train >=50 verified corrections。
- Supplemental protocol 满足预先固定的 Validation >=20 支持要求。
- 两侧均具有足够独立视频和冲突组。
- 完整状态及 RNG provenance PASS。
- 没有 GT/future inference leakage。
- 自然和困难负例齐备。

如失败：

`BLOCKED_DATA_SUPPORT`

不得进入正式模型训练。

输出：

- `PHASE9_SUPPLEMENTAL_CAPTURE_AUDIT.json`
- `PHASE9_CAUSAL_DATASET_MANIFEST.json`
- `PHASE9_CAUSAL_DATASET_REPORT.md`

---

# 4. PHASE B：建立真正的 Native Candidate Values 接口

这一阶段可以与 Phase A 并行开发和测试，但不得读取新的验证因果结果作为接口调参依据。

当前 Phase VIII 的：

`reproduction_tools/jev_phase8_candidate_submit.py`

主要实现了在研究性原生 replay 中提交显式合法 candidate pairs。

它没有证明神经网络输出的候选价值可以通过完整 deployed sliding inference 正确提交。

本阶段必须解决这个缺口。

## B1. 先审计 WWW 当前真实入口

重点检查：

- `gtr/modeling/meta_arch/gtr_rcnn.py`
- `GTRRCNN._apply_jev_match_decisions`
- `GTRRCNN.sliding_inference`
- `GTRRCNN.sliding_inference_GMT`
- `GTRRCNN.run_global_tracker_plus`
- `GTRRCNN.memory_bank`
- `GTRRCNN._jev_reactivation_action`
- `gtr/modeling/jev_assignment.py`
- `gtr/modeling/jev_runtime.py`
- `reproduction_tools/jev_phase8_candidate_submit.py`
- `reproduction_tools/jev_phase7_native.py`

明确实际的关联矩阵、候选顺序、全局求解、ID 提交、MEMORY 与 stale-bank 调用顺序。

不能为了实现新功能修改 GMT OFF 原始结果。

必须优先通过默认关闭、显式 opt-in 的方式集成。

## B2. 定义统一 Candidate Interface

输入包含：

- detection rows
- candidate track IDs（仅作为运行时引用）
- original GMT score matrix
- online state64
- candidate-relative evidence
- candidate legal mask
- NEW option
- actual conflict constraints

模型输出：

`candidate_values[Detection, Candidate]`

以及每个 detection 对应的 NEW value。

不得把绝对身份整数作为模型可学习特征。

不同候选集合长度必须被支持。

所有方法使用完全相同的 candidate representation 和 legal mask。

## B3. 统一 Global Assignment

设计一个共享的 constrained assignment/commit operator。

固定阈值、MLP、DeepSets、CAMEL-inspired model、JEV 必须调用同一实现。

现有历史身份不得被两个违反原生约束的检测重复占用。

需要为每个 detection 提供独立的 NEW dummy option，避免多个 NEW 竞争同一虚拟列。

原生跨视角共享身份规则必须保留，不能不加区分地将单摄像头一对一限制套用到所有跨相机事件。

严格处理：

- NaN / Inf score
- empty candidates
- all masked candidates
- duplicate IDs
- tied values
- zero detections
- zero active tracks
- high candidate count
- unavailable correct identity
- legal START_NEW
- stale-bank recovery after MATCH
- conflict-connected assignment groups

不要给某一个模型额外使用 GT 或更丰富的 online metadata。

## B4. 先做固定值控制

在训练任何候选模型前，使用 GMT 自身的原始候选分数作为 candidate values。

要求：

- 相同冻结输入产生确定输出。
- 可通过显式适配配置重现 GMT OFF 语义。
- 原始轨迹选择与实际提交的关联一致。
- 真实 production MATCH operator parity PASS。
- 全部 mutable state commit parity PASS。
- MEMORY/bank/birth/stale 状态正确。
- 下一帧真实 online state 与重新计算的 proposal 相容。
- 无未来信息或 GT。
- candidate permutation equivariance PASS。
- 所有方法共享相同求解器。

特别验证相同帧内多行冲突，不允许分别选取每一行最高分候选后才尝试随意修补冲突。

注意，Phase VIII 已通过的 production MATCH helper 单元测试不能代替这里的完整部署接口验收。

## B5. 集成要求

允许在新分支中新增：

- `gtr/modeling/jev_candidate_assignment.py`
- `gtr/modeling/jev_candidate_policy.py`
- `gtr/modeling/jev_candidate_features.py`
- `reproduction_tools/test_jev_phase9_candidate_native.py`

建议文件名可调整，但必须将：

1. Candidate scoring
2. Assignment
3. Native state commit

拆为明确的独立接口。

不能将研究脚本中一次性 monkey-patch 的求解器当作正式生产实现。

输出：

- `PHASE9_NATIVE_CANDIDATE_INTERFACE.md`
- `PHASE9_CANDIDATE_SUBMIT_PARITY.json`
- `PHASE9_CANDIDATE_INTERFACE_TESTS.json`

接口门槛失败：

`BLOCKED_DEPLOYMENT_CONTRACT`

禁止进入正式闭环架构比较。

---

# 5. PHASE C：数据可学习性 Sanity Check

只有 A/B 门槛都通过，才运行此阶段。

先选择少量已经核验、身份确切且存在多个合法候选的训练事件。

首先使用简单 Candidate MLP。

随后使用 DeepSets 和最简 Candidate JEV。

必须证明：

- 可靠的训练标签可以被拟合。
- 候选 ID 排列变化不影响语义结果。
- 模型能在有正确候选时选中正确候选。
- 原 GMT 正确时模型不会总是错误改配。
- NEW 能正常学习。
- masked candidates 的概率严格为零。
- 不存在训练与在线特征归一化不一致。
- 全部训练 loss 有限，梯度有限。

Tiny-set overfit 不是泛化证据，只证明输入、标签、网络和优化链路能够正常工作。

如果三个模型都无法过拟合小数据：

先检查特征分离度、anchor、GT 对齐、候选映射、mask、标签冲突和数据归一化。

不得立即增大参数或者开始长时间训练。

输出：

`PHASE9_TINY_OVERFIT.json`

失败则停止并明确报告原因。

---

# 6. PHASE D：三个候选模型的公平训练

主要比较：

## D0 — 普通强基线

- GMT OFF
- Fixed association rule
- Learned state-conditioned threshold
- Bidirectional candidate score normalization
- Frozen B2 MATCH-only

## D1 — Candidate MLP

输入：

相同 state64 + candidate12 + mask + online candidate context。

模型直接输出每条合法候选的值和 NEW 值。

不是先把所有候选压缩成一个单独 score，再给 JEV 一个更大的输入。

## D2 — Candidate DeepSets

使用可变长度候选集合。

候选表示使用共享编码器和 masked pooling 建模上下文。

必须验证排列等变性。

训练数据、损失、计算预算与 JEV 保持公平。

## D3 — Candidate-Conditioned JEV

保留 Question/Action-conditioned 的结构化思路。

加入：

- detection-to-candidate evidence
- candidate competition context
- legal candidate masking
- semantic NEW
- correctness estimation
- long-horizon action utility estimation

优先实现轻量模型，而不是直接构建大型 Transformer。

以原 B2 约 34K 参数为一个容量参照，同时提供参数/计算相近的 MLP 与 DeepSets。

如参数严格一致导致架构不合理，记录实际参数、MACs、前向延迟，并同时报告参数匹配与计算匹配对照。

不得以更大的 JEV 对比刻意弱化的 MLP。

## D4 — 两层监督消融

这是证明 JEV 研究价值的核心。

同一网络分别训练：

1. Current candidate correctness。
2. H32 causal utility。
3. Correctness + H32 utility。
4. 预先冻结的 candidate ranking/advantage 监督。

分析：

- 正确身份 Rank-1
- Candidate MRR
- Corrective Recall
- Wrong-correction Rate
- False Birth / Merge
- Utility Regret
- Native closed-loop AssA/HOTA

如果因果监督优于普通 correctness，也不能直接将全部收益归因于 JEV 架构；必须让 MLP 和 DeepSets 使用相同的因果监督进行比较。

## D5 — 数据规模、epoch、容量与归一化

所有实验按预注册方案执行。

数据规模：

25% / 50% / 75% / 100%，按独立事件组采样。

训练轮数：

5 / 20 / 50 / 100 epochs。

模型容量参考：

8K / 34K / 128K / 500K 参数。

输入处理：

raw / train-stat standardized / LayerNorm。

首先根据 tiny-overfit 和初始公平结果判断是否需要完整曲线，避免所有组合无条件笛卡尔积运行。

所有参与架构比较的模型使用同一训练预算和冻结的选择协议。

动态候选 Top-K 必须根据 TRAIN 的候选召回/速度权衡预注册，不得默认 Top-8，也不得通过 GT 在运行时保留正确候选。

保存候选 recall@8/16/32/64/full，并检查是否排除 rank19/27/45 等真实纠错候选。

## D6 — 训练组织

至少分开统计：

- Natural samples
- Hard corrections
- Hard negatives
- Unknown/censored

训练可使用分层采样。

校准和最终自然性能报告必须考虑真实事件频率及采样偏差。

Unknown/censored 不能被强制解释为错误候选。

对没有执行过 counterfactual branch 的候选，不得构造假效用标签。

输出：

- `PHASE9_TRAINING_ABLATION.json`
- `PHASE9_DATA_SCALING.json`
- `PHASE9_CAPACITY_ABLATION.json`
- `PHASE9_SUPERVISION_ABLATION.json`
- `PHASE9_NORMALIZATION_ABLATION.json`
- `PHASE9_MODEL_COMPARISON.md`

---

# 7. PHASE E：真正在线闭环与独立验证

只有 A、B、C、D 通过，才开启预留 controller-heldout：

- video20
- video21
- video22

禁止将这些视频用于新结构设计、损失权重调参或 checkpoint 选择。

所有方法必须：

- 使用同一 frozen GMT perception。
- 使用相同可用候选和状态字段。
- 使用相同 shared assignment/commit operator。
- 相同 birth/memory/stale 语义。
- 每一帧重新计算真实 online state。
- 不使用旧 OFF 的未来动作缓存。
- 不使用 GT 做运行时决策。

主要指标：

- HOTA
- AssA
- IDF1
- IDSW
- MOTA
- Frag

额外记录：

- Correction N01/N10
- Wrong-correction Rate
- False Merge
- False Birth
- Correct/Wrong ID Duration
- Candidate Oracle Recall
- Per-group Causal Utility
- Online Memory Contamination
- Cross-camera Identity Continuity
- Latency p50/p95
- GPU Memory Overhead

真实 pooled TrackEval 由全部完整视频合并运行，不能用简单平均替代。

按照 Phase VIII 的已有架构 Gate，重点审查：

- JEV 是否比最强预注册普通关联基线高至少 0.1 pooled HOTA。
- 是否高至少 0.2 pooled AssA。
- 至少两个 heldout 视频改善。
- 纠错增益未被反改错抵消。
- p95 新增关联延迟不超过 10 ms。
- 额外 GPU memory 不超过 128 MiB。

同时报告原始序列结果、多种子稳定性及统计不确定性。

这些门槛只是预注册的研究操作点，不代表通过后已经有统计显著性或具备 WWW 中稿保证。

如果 JEV 不优于强 MLP/DeepSets，必须报告 `NO_GO_JEV_ARCHITECTURAL_ADVANTAGE`。

不要因为总体 HOTA 接近就隐瞒罕见纠错事件上的成功或退化。

也不要因为某个困难子集表现更好，就忽略完整视频自然分布的负结果。

---

# 8. PHASE F：为三阶段 Unified JEV 保留下一步接口

我们的长期目标依然是：

MATCH — WHO?

MEMORY — WRITE?

REACTIVATION — OLD OR NEW?

本阶段不要求无条件训练三个模块。

MATCH 通过独立闭环 Gate 后，才开始专门的 MEMORY/REACT 科学实验。

MEMORY 必须先证明真实写入改变后续实际身份查询结果，而不只是 gallery hash 或 cosine score 变化。

REACTIVATION 必须先验证 native relative hook 与候选支持，不能将没有经过生产核验的研究性恢复模型包装为完整线上系统。

最终 Unified JEV 要证明：

- 共享结构优于独立控制头。
- MEMORY/REACT 对 MATCH-only 有可验证增量。
- 三者共同减少长期身份错误传播。
- 在同等数据/预算下优于强普通网络。

没有证据时不进行 Unified 联合训练。

---

# 9. 严格的停止条件

### G_DATA

补充数据核验失败，或自然/困难负例缺失：

STOP / BLOCKED_DATA_SUPPORT

### G_NATIVE

候选评分不能经过生产全局求解和实际提交正确影响下一帧：

STOP / BLOCKED_DEPLOYMENT_CONTRACT

### G_TINY

可靠的少量监督不能被正常拟合：

STOP / BLOCKED_LABEL_OR_OPTIMIZATION_CONTRACT

### G_ARCH

JEV 在同数据、同预算下不能优于强普通网络：

NO_GO_JEV_ARCHITECTURAL_ADVANTAGE

### G_HELDOUT

独立闭环关联质量、纠错收益、稳定性或在线效率未通过：

NO_GO_CLOSED_LOOP_GENERALIZATION

### G_LIFECYCLE

MEMORY/REACT 缺少有效监督或生产接口：

BLOCKED_UNIFIED_LIFECYCLE

禁止根据负结果修改既有 gate，再声称原实验通过。

本次任务内：

`FULL24_AUTHORIZED=false`

`OFFICIAL_TEST_AUTHORIZED=false`

不得提前启动 Full24 或百万级正式数据生产。

---

# 10. 存储与执行约束

GPU 资源使用现有服务器环境，优先空闲设备，不干扰其他任务。

复用已有 GMT checkpoint、VisionTrack 数据与 perception cache。

新实验优先放在独立运行目录，避免重复复制大型原始数据。

GitHub 只提交：

- 必要源码
- 单元与回归测试
- 实验配置
- 分割规则与采样清单
- 论文审计
- compact JSON 指标
- Markdown 报告
- 小型诊断样本
- SHA/provenance manifest

大型原始快照、缓存、完整 fork logs 和 checkpoint 保留服务器端，并明确存储路径和 SHA。

所有有界任务支持 resume 和原子完成 manifest。

任何历史 checkpoint 删除都必须有独立依赖证明，不能基于目录名或时间直接删除。

不得使用无范围的 `rm -rf`。

不得为了降低 wall-clock 时间跨越科学 Gate。

---

# 11. 最终交付

创建：

`docs/PHASE9_RESEARCH_GOAL.md`

`docs/PHASE9_EXTERNAL_CODE_AUDIT.md`

`docs/PHASE9_METHOD_DIFFERENTIATION.md`

`docs/PHASE9_SUPPLEMENTAL_CAPTURE_PROTOCOL.md`

`docs/PHASE9_CAUSAL_DATASET_REPORT.md`

`docs/PHASE9_NATIVE_CANDIDATE_INTERFACE.md`

`docs/PHASE9_ARCHITECTURE_DESIGN.md`

`docs/PHASE9_FINAL_RESEARCH_REPORT.md`

以及：

`reports/JEV_PHASE9/EXTERNAL_REPOSITORY_MANIFEST.json`

`reports/JEV_PHASE9/SUPPLEMENTAL_CAPTURE_AUDIT.json`

`reports/JEV_PHASE9/CAUSAL_DATASET_AUDIT.json`

`reports/JEV_PHASE9/CANDIDATE_INTERFACE_TESTS.json`

`reports/JEV_PHASE9/NATIVE_COMMIT_PARITY.json`

`reports/JEV_PHASE9/TINY_OVERFIT.json`

`reports/JEV_PHASE9/BASELINE_COMPARISON.json`

`reports/JEV_PHASE9/ARCHITECTURE_ABLATION.json`

`reports/JEV_PHASE9/HELDOUT_RESULTS.json`

`reports/JEV_PHASE9/FINAL_GO_NO_GO.json`

对于没有资格执行的实验，状态必须为 NOT_RUN/BLOCKED，不得创建没有实际结果的伪指标。

每个阶段都要写：

WHAT DID WE LEARN?

最后明确回答：

1. 旧 JEV 的效果为什么会被简单规则超过？
2. 补充监督是否解决了候选纠错样本不足？
3. Candidate MLP 是否已经足够？
4. DeepSets 或 CAMEL-inspired context 是否能解释全部关联收益？
5. JEV 的 Question/Action 结构是否产生了独立贡献？
6. 长期因果 utility 是否比普通正确性标签更有价值？
7. 当前最值得保留的论文创新到底是架构、训练方法、决策机制，还是因果数据构建？

如果 JEV 失败，必须真实汇报，并据此提出最小的下一步假设，不允许无限扩大模型和重训来寻找正向结果。

**FINAL GOAL：用真实候选、真实原生状态和公平对照，证明或证伪 Candidate-Conditioned Causal JEV 是否具有独立的在线身份关联价值，并为最终 MATCH–MEMORY–REACTIVATION Unified JEV 建立可靠基础。**

从 Phase 0 自主执行。通过 Gate 就进入下一阶段，失败就保存完整证据、定位具体原因并终止不合格的后续阶段。
