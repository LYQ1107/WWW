# Codex 最终研究执行任务：GMT-JEV 双问题身份验证与快速突破

## 0. 最高优先级：立即执行，不要只给计划

你现在负责 `LYQ1107/WWW` 的下一轮 WWW 研究任务。

**当前日期：2026-10-15。目标：WWW 2027。**

研究仓库：https://github.com/LYQ1107/WWW

当前研究分支：

`research/gmt-jev-risk-association-20261014`

最近已核验远程提交：

`fa98ff7ca204196fe2c4a6cbad180e9cef8f6f5c`

现有工作目录：

`/home/liuyeqiang/WWW_gmt_jev_20261014`

现有运行目录：

`/home/liuyeqiang/WWW_gmt_jev_runtime/20261014_v1`

请首先 `git fetch`，核对最新远程 HEAD。以上路径和提交是已知参考，不是可以跳过检查的事实。

本次任务不是再次设计一套完整 JEV，也不是继续堆叠 Question Reader，而是：

**在保留完整 GMT 和已经取得正向结果的 MLP 的基础上，验证一个真正具备 JEV 式问题条件化身份决策能力的轻量验证器。**

必须完成工程实现、训练、受控评测、效率分析、研究 Goal 修订和 GitHub 推送。

不要在完成一个局部实验后直接结束整个任务；应按下述预注册的分支条件推进。但不得突破预算、数据资格或科学停止条件。

---

# 一、正式修订 FINAL GOAL

## 1.1 新的科学目标

建议研究名称：

**GMT-JEV: Question-Conditioned Identity Verification for Efficient Multi-Camera Multi-Target Tracking**

核心研究目标：

在完整 GMT 的 GTA Transformer、VFCE/RPCE、原生合法匹配及 Gallery/Bank 机制基础上，引入轻量、问题条件化的身份验证模块。

该模块不重新学习完整视觉关联，不替代 GMT 的 GTA，也不直接预测新的 Global ID。

它必须回答两个不同且具有明确决策作用的问题：

**Q1：当前候选 Global ID 的历史身份依据可靠吗？**

**Q2：在同一原生状态下，应该保持 GMT 的关联决定，还是接受轻量 MLP 给出的合法关联修改？**

让这两个问题共享当前多摄像头身份状态，并直接参与最终合法决策。

科学目标包括：

1. 保留 GMT 完整系统的强基线。
2. 保留现有 MLP 已取得的开发集正向结果。
3. 验证 JEV 式问题条件化决策能否在普通 MLP 之上产生独立增益。
4. 在提高 HOTA、AssA、CVIDF1 的同时，避免增加身份切换、碎片化和长时间错误关联。
5. 将新增模型的完整图像推理开销控制在合理范围。
6. 与普通 MLP 验证器、Attention 验证器及简单置信度策略进行公平比较。
7. 最终结果必须具备可复现性和合格的独立评价证据。

这里不要求复刻 JEV 的所有结构。

**如果 JEV 风格的组件没有经过消融证明有用，不允许仅凭命名将其写成论文创新。**

## 1.2 Goal 文件修改

读取：

- `reports/GMT_JEV/MASTER_GOAL.json`
- `reports/GMT_JEV/GO_NO_GO.json`
- `reports/GMT_JEV/LEARNED_GATE_RESULTS.json`
- `docs/GMT_JEV/FINAL_RESEARCH_REPORT_ZH.md`
- `docs/GMT_JEV/LEARNED_GATE_RESULTS_ZH.md`
- `CURRENT_RESEARCH_GOAL.md`

新建分支：

`research/gmt-jev-typed-verifier-20261015`

更新现有 Master Goal，建立带版本号的 V2 研究修订。

新增：

`docs/GMT_JEV_VERIFIER/FINAL_GOAL.md`

`reports/GMT_JEV_VERIFIER/GOAL.json`

并更新权威当前 Goal 入口，使后续 Codex 不会错误地重启旧 CIC 或旧 Recovery 任务。

必须保留历史证据：

- Original GMT 完整系统；
- Strong Fixed；
- R2/R3 Recovery NO_GO；
- JEV Fastcheck NO_GO；
- GMT + CIC-JEV NO_GO；
- GMT + MLP 开发集正向结果。

新 Goal 中明确：

`BASE_SYSTEM = COMPLETE_ORIGINAL_GMT`

`POSITIVE_REFERENCE = GMT_PLUS_MLP_STEP300`

`JEV_VERIFIER = HYPOTHESIS_PENDING`

`SCIENTIFIC_SUCCESS = false`

`INDEPENDENT_VALIDATION = PENDING`

不要修改旧结果，不要把历史失败变成成功。

先提交一次 Goal 修订并推送，然后进入算法执行。

---

# 二、冻结现有正向结果，绝不能把它弄丢

当前必须保留的历史开发结果：

评测视频：

VisionTrack TRAIN16

1034 个场景时刻、2068 张图像、两台摄像头。

主要协议为完整原生 raw tracking。

| 方法 | HOTA | AssA | CVIDF1 | IDSW |
|---|---:|---:|---:|---:|
| Original GMT | 68.054089 | 67.064493 | 86.544619 | 1898 |
| GMT + MLP | 68.654661 | 68.098573 | 88.001081 | 1945 |
| GMT + Attention | 67.508007 | 65.804120 | 85.595046 | 1965 |
| GMT + CIC | 67.365381 | 65.526387 | 85.699803 | 2031 |

MLP 相对 GMT：

HOTA +0.600573

AssA +1.034080

CVIDF1 +1.456462

但 MLP 的 IDSW 增加 47。

真实完整图像 FPS：

GMT = 3.409768

MLP = 3.129765

CIC = 2.396665

已有 MLP 新增可训练参数仅 5,119 个。

这些结果是当前唯一可信的正向开发起点，不能丢失。

相关 checkpoint 的确切路径和 SHA 请从：

`reports/GMT_JEV/LEARNED_GATE_SMOKE_TRAINING.json`

`reports/GMT_JEV/LEARNED_GATE_RESULTS.json`

读取，而不是重新训练一个不同的 MLP。

原始 GMT checkpoint 已知位置：

`/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth`

SHA256：

`cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8`

先检查文件实际存在、SHA 一致、环境可加载。

原始 GMT 包含约 53,515,309 个参数，保留完整 VFCE1024、RPCE128 和 GTA。

必须先复用现有基线严格等价证明：

`reports/GMT_JEV/BASELINE_PARITY.json`

不能重新切换到 Strong Fixed。

不能将旧 JEV_DIRECT 的预测当作 GMT。

不能使用不同前端的历史成绩与本实验直接计算提升。

---

# 三、P0：先验证 JEV 是否有值得学习的决策问题

**这一阶段是整个任务最重要的前置条件。**

不要一上来就训练 Question Head。

现有报告已经暴露一个严重限制：

TRAIN12/13/14 中：

- 26,297 条具有唯一认证合法身份的训练行；
- 原 GMT 已认证错误行只有 97 条；
- 当前有界动作中，仅 17 条拥有离线纠错见证。

这意味着能可靠监督“修改 GMT 关联是否正确”的样本可能非常少。

另外，当前 MLP 虽然提高了 HOTA，但其第一个真实原生身份分歧：

GMT 创建 ID52；

MLP 接受 ID25；

该检测的离线 GT 属于 UNKNOWN。

因此不能把它称为已经证明的 JEV 或 MLP 身份纠错。

## 3.1 必须执行同状态决策比较

在已获准 TRAIN12/13/14 上，加载：

Original GMT

Frozen MLP step300

使用同一个原始 GMT pre-commit 状态：

1. 计算原始 GTA 聚合关联分数。
2. 计算 MLP 修正后的关联分数。
3. 分别使用同一原生合法匹配约束求解。
4. 比较原始方案与 MLP 方案。
5. 不要因为 MLP 改变了关联就默认它是正确的。
6. 用离线 GT 对两套合法方案进行认证。
7. 标记是否能够确定哪个方案更好。
8. 未知时保持 UNKNOWN。

必须严格区分：

- 原始 GMT 正确、MLP 错误；
- 原始 GMT 错误、MLP 正确；
- 两者都正确或等价；
- 两者都错误；
- 两者的合法动作都无法认证；
- 正确身份不在当前合法候选中；
- 多个预测 ID 对应同一真实人物；
- 原始 Bank/新生身份的结果不确定。

不要用不同策略各自运行到未来后形成的不同 Global ID 数字直接比较。

必须首先在同一原生状态上作可认证的局部决策比较。

记录：

`PAIRED_PLAN_OPPORTUNITY.json`

`DECISION_LABEL_ELIGIBILITY.json`

`CURRENT_IDENTITY_CONFLICT_AUDIT.json`

## 3.2 历史可靠性标签

只使用当前匹配之前已经发生的历史观测。

Q1 的目标不是当前目标与身份是否匹配，而是该身份历史本身是否可靠。

建议三种标签：

**RELIABLE**

过去已有足够有效观测，且可认证身份归属一致，没有已确认的多人混杂。

**CONFLICTED**

历史中出现具有足够证据的不同真实人物混合，或可认证的跨摄像头身份归属冲突。

**UNKNOWN**

已知历史不足、身份归属含糊、多个原生 ID 别名未解决，或无法可靠认证。

需要尽量复用已有纯度/覆盖率审计标准。

对于未达到可靠标注条件的事件，不允许为了形成三分类平衡而强行贴上 RELIABLE 或 CONFLICTED 标签。

GT 只能用于训练监督和离线分析。

推理阶段只允许使用已有 Gallery、真实历史特征、相机编号、原生关联分数等因果输入。

严禁使用未来帧来判断当前历史是否可靠。

## 3.3 Q2 的监督

Q2 只能学习在同一个 GMT 原生状态下：

`KEEP_GMT`

或

`APPLY_MLP`

哪种合法方案更合理。

如果两个方案相同，无须学习一次虚假的选择。

如果 GMT 和 MLP 的差异不可认证，不能标注 MLP 正确或 GMT 正确。

如果两个方案都不正确，不能强行选择较好但未经认证的一个。

如果方案合法性或原生阈值条件不成立，不能作为正常正例。

请按人、视频、摄像头、时间块去重或分组。

一个持续多帧的错误不能当作大量独立训练正例。

## 3.4 P0 的准入门槛

必须输出：

- 可认证 KEEP_GMT 样本数；
- 可认证 APPLY_MLP 样本数；
- 有效独立人物/时间依赖组数；
- Q1 的三类有效支持量；
- UNKNOWN 占比；
- 合法候选覆盖率；
- 同一人物多 ID 别名比例；
- 训练样本是否跨越原有时间留出块；
- Q2 是否存在非零梯度的真实监督。

如果 Q2 可靠标签极少，优先检查已有授权 TRAIN 历史和既有日志是否有可复用的有效事件。

不得因此建立新数据集或反复伪造标签。

如果没有可学习的 Q2 监督，不允许直接启动大训练。

应执行本提示词后面规定的有界失败转向，而不是无限增加数据或模型。

---

# 四、P1：真正实现双问题 JEV Verifier

新建：

`gtr/modeling/gmt_jev/typed_verifier.py`

以及必要单元测试。

现有：

`gtr/modeling/gmt_jev/correction.py`

必须保留并兼容。

本轮不改写已有 MLP 的输出和训练权重。

## 4.1 正确的推理结构

原始完整 GMT：

Detector
  ↓
VFCE / RPCE
  ↓
GTA Transformer
  ↓
Original Global ID scores
  ↓
Original native legal assignment

同时使用现有 MLP：

Original scores
  ↓
Frozen MLP residual
  ↓
MLP-corrected Global ID scores
  ↓
同一原生合法 assignment

当原始 GMT 与 MLP 方案存在实际差异时：

读取共同的 pre-commit identity state。

进入：

JEV-inspired Typed Verifier

Q1：
当前争议 Global ID 的历史证据是否可靠？

Q2：
保留 GMT 原方案，还是采用 MLP 原方案？

输出：

`KEEP_GMT`

或

`APPLY_MLP`

然后执行被选择的完整合法分配方案。

最后仅进行一次原生身份提交。

保持原有：

Gallery、Bank、REACT、WRITE、ID 创建规则。

不能引入另一个独立身份管理器。

不能额外创造不受原生候选约束的 Global ID。

如果完整匹配方案包含多个检测，选择必须按整个合法方案进行，不能逐行拼接两个方案造成身份占用冲突。

如果原生 Bank、出生或阈值机制使两套方案并非可直接替换，必须先证明实际接口契约，不能通过手工修改 `track_ids` 假装实现。

## 4.2 Question 1：Identity Evidence Reliability

输入：

- 争议 Global ID 的同相机历史；
- 异相机历史；
- 同/异相机外观一致性；
- 历史数量；
- 当前身份年龄；
- GMT 原始支持；
- 现有 MLP 相关证据；
- 已提交历史之间的冲突。

不需要大型视频 Encoder。

优先复用当前 MLP 的24维证据及已计算的跨摄像头特征。

允许使用轻量历史摘要，但必须来自实际已提交状态。

输出：

- 历史可靠性分布；
- 可供 Q2 直接使用的可靠性表示。

Q1 的结果必须真实进入 Q2 的判断路径。

不能只计算一个辅助分类准确率。

不能像旧 JEV 一样：辅助头预测正确，但 logits 对最终结果没有作用。

## 4.3 Question 2：Should We Trust the Correction?

输入：

- Q1 的历史可靠性结果；
- Original GMT 的完整合法匹配方案；
- Frozen MLP 的完整合法匹配方案；
- 两方案的差异；
- 受影响身份的跨摄像头支持；
- 当前候选竞争与不确定性。

输出：

`KEEP_GMT`

`APPLY_MLP`

使用明确的类型化输出，不再只是计算一个无语义的边残差。

如果证据不足，模型可以选择保守回退到 GMT；但不能把所有 UNKNOWN 一律标记为“GMT 一定正确”。

对不确定事件必须保留其不确定性语义，避免把它们变成训练伪标签。

推理时模型只使用已经观测到的状态。

GT 和任何事后评价结果不得进入模型。

## 4.4 共享状态与实现限制

建议实现：

Shared identity encoder
+
Q1 typed reliability reader
+
Q2 typed plan decision reader

首先采用极小网络。

初步目标：

新增参数量 ≤ 8,000。

如确实需要放宽到10,000以上，必须先提供必要性证据。

不要采用256维、十几层 Attention 的旧完整 JEV。

优先使用32维左右的轻量共享表示。

尽可能向量化。

只有 GMT/MLP 发生真实合法方案差异时才启用验证计算；如需每帧更新历史摘要，要求复用已有 MLP 状态，而不是重复计算余弦矩阵。

不要给每个 Global ID 额外保存大量图像历史。

不要产生新的持久大型缓存。

不要为了减少参数而增加大量 Python 循环、GPU/CPU 同步或重复 Hungarian 求解。

## 4.5 严格回退契约

增加控制选项：

`verifier_enabled=False`

或等价 OFF 模式。

要求：

Verifier OFF = 原 GMT + MLP 完整行为完全一致。

不只是 HOTA 接近，而是：

- 关联评分一致；
- 合法匹配一致；
- Global ID 提交一致；
- Gallery/Bank 一致；
- 完整预测一致。

另外，强制 `KEEP_GMT` 时必须返回相同原生状态下的 GMT 匹配结果。

强制 `APPLY_MLP` 时必须返回相同状态下的 MLP 匹配结果。

零增量初始化时默认保持现有 MLP 行为，不得因新增未经训练的验证器就损坏已有的正向模型。

只有训练出真实可验证决策后，才允许其改变选择。

测试所有选项在没有候选、只有一个候选、候选冲突、相机切换、Bank 恢复和首次跨相机初始化等情况下的合法性。

---

# 五、P2：先确认两个 Question 真正发挥作用

必须进行以下机制测试：

**测试 A：Q1 输出是否影响 Q2？**

在相同状态下干预 Q1 的输出，保持其他输入不变。

检查 Q2 分布是否发生符合预期的变化。

如果没有变化，不能宣称 Q1 影响最终决策。

**测试 B：Q2 是否影响 native commit？**

在一个合法的有分歧测试状态中，分别强制 KEEP_GMT、APPLY_MLP。

必须观察最终原生 Global ID 提交不同，并符合各自完整合法方案。

**测试 C：两问题是否共享同一个历史状态？**

证明 Q1 和 Q2 读取共同的历史身份信息，而不是互相独立的两个普通 MLP。

**测试 D：问题是否可辨识？**

检查 Q1 对历史可靠性、Q2 对关联动作的监督梯度。

不能因为加了两个 Head 就默认模型学到了两种不同能力。

**测试 E：防止破坏 GMT/MLP**

Verifier OFF 全视频预测必须与已冻结 MLP 一致。

原始 GMT OFF 基线也保持一致。

**测试 F：实时性**

同一 pre-commit 状态下记录：

- 共享特征读取开销；
- Q1 开销；
- Q2 开销；
- 是否额外调用求解器；
- 对真实提交的额外开销。

任何 hook 或状态替换必须能在异常发生后正确恢复。

---

# 六、P3：同监督、同动作空间的公平学习实验

不要先训练复杂版本。

最小实验组：

**B0：Original GMT**

主要基线。

**B1：GMT + Frozen MLP**

当前最佳开发集正向方法。

**B2：GMT + Frozen MLP + Ordinary Verifier**

使用普通 MLP 验证器。

输入与 JEV 相同。

能够选择 KEEP_GMT / APPLY_MLP。

不使用类型化 Question 结构。

**B3：GMT + Frozen MLP + Typed JEV Verifier**

本轮主要候选。

必须与 B2：

- 输入相同；
- 可观察历史相同；
- 候选合法性相同；
- Q1/Q2 有效监督相同；
- 最终可执行动作相同；
- 训练样本相同；
- 训练顺序相同；
- Loss 相同或只包含明确说明的结构差异；
- 训练步数相同；
- 可训练容量相近；
- 原生求解器相同。

此外保留：

`GMT + MLP + simple confidence verifier`

作为无训练或低成本规则对照。

不能用一个很弱的普通 Verifier 来证明 JEV 先进。

## 6.1 最小 Loss

建议：

`L = L_decision + λ1 L_reliability + λ2 L_preserve + λ3 L_regularize`

其中：

**L_decision**

监督在可认证状态中选择 KEEP_GMT 或 APPLY_MLP。

如果两者都无法认证，不产生虚假的动作标签。

**L_reliability**

监督 Q1 对既有身份历史可靠性的判断。

可以采用掩码分类损失，UNKNOWN 必须正确处理。

**L_preserve**

对原本正确且已有充分认证的 GMT/MLP 方案，惩罚错误改动。

**L_regularize**

限制无证据的大规模干预，控制策略的复杂度。

Loss 权重在新实验启动前冻结。

不得根据 TRAIN16 的正负结果反复调权重。

普通 Verifier 必须获得公平的监督目标。

## 6.2 快速训练预算

优先复用：

现有完整 GMT checkpoint；

现有 MLP step300 checkpoint；

TRAIN12/13/14 已获准的数据；

现有原生推理入口和离线监督工具。

GMT 冻结。

MLP 冻结。

只训练新增小验证器。

先执行：

- 无梯度工程测试；
- 监督资格审计；
- 非零梯度 smoke；
- 固定300步训练。

只要已有监督不足以支持动作学习，就不要启动300步模型训练。

不要因为还有 GPU 预算就扩展到1000或20000步。

禁止再从零训练 Stage1、GTA 或原来的 JEV。

训练时可以即时从现有原生状态构造样本。

不保存新的完整视频数据集。

样本 RAM 缓冲建议不超过32MiB。

只保存模型权重、配置、关键统计和实验哈希。

## 6.3 学习过程中必须记录

每25或50 steps 记录：

- 总 Loss；
- Q1 Loss；
- Q2 Loss；
- 正负监督事件数；
- 有效人物/时间依赖组数；
- UNKNOWN 比例；
- 被 Mask 的比例；
- 梯度范数；
- 两个 Head 是否都得到有效梯度；
- Verifier 与当前 MLP 的动作分歧数；
- 学习式策略实际可能改变的合法计划数。

不能只记录训练 Loss。

保存固定300步终点 checkpoint。

不根据 TRAIN16 选择最优 step。

---

# 七、P4：必须有真正的完整在线效果

当前开发评测范围：

TRAIN16，1034场景帧，双摄像头。

该视频此前已经暴露，只能作为开发诊断。

必须以相同冷启动、同样完整 GMT、同样前端、相同原生评测器运行：

1. Original GMT；
2. Frozen GMT + MLP；
3. GMT + MLP + ordinary verifier；
4. GMT + MLP + typed JEV verifier。

没有必要重新跑已经有完整、输入与源码SHA完全对应的冻结参考；能严格复用的直接复用。

但新模型必须进行完整原生在线评测，不能只使用固定状态缓存替代。

### 7.1 核心指标

必须输出：

HOTA、AssA、DetA、IDF1、CVIDF1、CVMA、MOTA、IDSW、Frag。

还要统计：

- Verifier 总调用次数；
- GMT/MLP 实际存在分歧的次数；
- Q1 判定历史冲突的次数；
- Q2 选择 KEEP_GMT 的次数；
- Q2 选择 APPLY_MLP 的次数；
- 认证 N01；
- 认证 N10；
- UNKNOWN 修改；
- 新生 ID 数量；
- 额外身份碎片；
- 错误身份合并；
- 长时间 Global ID 不一致；
- 每类事件真实提交后的在线变化。

注意：
不同策略未来的身份状态可能不同。

不能将不同策略运行的 ID 编号直接当成同一个身份。

### 7.2 预注册成功目标

历史开发参考：

GMT HOTA = 68.054089。

GMT + MLP HOTA = 68.654661。

首先，JEV Verifier 必须守住已取得的 MLP 正向结果。

然后争取满足以下两种增量条件之一：

**路径 A：质量全面提升**

HOTA 相对 MLP 至少 +0.20 点；

AssA 与 CVIDF1 不低于 MLP；

IDSW 和身份碎片不恶化。

**路径 B：身份证据更稳定**

HOTA 相对 MLP 下降不超过0.05点；

AssA、CVIDF1 不明显下降；

IDSW 比 MLP 至少降低5%；

历史身份不一致持续量也确有改善。

无论哪一种路径，都必须仍然超过同条件 GMT。

以上仅作为快速开发阶段的预注册目标。

不能将单种子 TRAIN16 正向结果称为独立科学成功。

如果普通 Verifier 同样达到或超过 JEV：
必须承认 JEV 结构独立优势没有建立。

如果新方法在 HOTA 上有提升，但产生更多出生或更严重身份碎片：
不能只看 HOTA 宣称突破。

### 7.3 对首次身份分歧进行严格分析

特别复核当前已知：

GMT 创建 ID52；

MLP 采用 ID25；

此事件离线 GT 是 UNKNOWN。

新 JEV 选择该事件时，必须准确报告它选择了什么。

不能把 UNKNOWN 事件事后改为正确身份。

如果新 Verifier 的改进主要来自 UNKNOWN 的不同处理，必须如实披露。

如果认为一次身份修改导致了后续长时间变化，只有在实际执行受控配对原生干预、恢复相同状态并继续完整运行后，才能将其称为因果证据。

否则只称为回顾性关联。

---

# 八、P5：必须进行结构独立性验证

如果 Typed JEV 在开发集取得正向增益，立即进行最小消融：

**A. 去除 Q1 历史可靠性输入。**

检查 HOTA/AssA/CVIDF1 和真实关联选择是否改变。

**B. 去掉类型化问题结构。**

换成容量匹配的普通 MLP Verifier。

**C. 去除跨摄像头证据。**

其他输入及训练保持不变，判断跨相机历史是否必要。

**D. 只使用 Q1 的可靠性，不使用 Q2 的问题条件化决策。**

检查增益是否只是简单风险阈值。

**E. 仅使用 MLP，不使用 Verifier。**

与已有正向模型直接比较。

只有在真实消融中被证明有效的机制，才可以作为论文贡献。

特别注意：

Q1 输出与 Q2 决策直接相连，只是工程必要条件，不等于已经证明 Q1 是算法增益原因。

普通 Attention、普通 MLP 也可以表达多种决策。

我们必须用公平实验而不是术语命名证明差别。

---

# 九、P6：实时性和部署开销

当前同GPU7真实完整图像参考：

GMT = 3.409768 FPS。

GMT + MLP = 3.129765 FPS。

MLP 已经低于预注册95%效率门槛。

因此，不允许新 Verifier 再明显拖慢系统。

目标：

`FPS_new >= 0.95 * FPS_GMT`

即在原测量条件下争取：

`FPS_new >= 3.239`

但应使用本轮同GPU、同环境下重新核对的基线来正式判断。

重点优化：

1. 不重复构建24维跨相机特征。
2. 共享冻结 MLP 已有的历史摘要。
3. 仅在真实合法方案分歧时调用 Verifier。
4. 避免对所有候选运行独立 Python 循环。
5. 避免新增不必要 CUDA 同步。
6. 避免计算完全重复的历史向量归一化。
7. 避免额外创建新的全量 Gallery。
8. 不用新的大模型替换 GTA。
9. 检查是否因重复求解同一 assignment 增加开销。

注意：之前尝试的特征缓存没有加速，不能默认缓存一定更快。

任何新优化必须先证明：
完整 native 预测与优化前严格等价。

然后才进行速度测试。

记录：

- 原始 GMT FPS；
- Frozen MLP FPS；
- 普通 Verifier FPS；
- Typed JEV FPS；
- 额外参数；
- Stage2 p50/p95；
- Verifier 独立耗时；
- 特征构建耗时；
- 峰值显存。

完整 FPS 必须包含真实图像、检测、VFCE/RPCE、GTA、匹配和原生身份更新。

固定特征回放不能冒充完整 FPS。

如果 JEV 有开发精度提升，但速度未达标：
保留质量候选，同时标记效率门槛失败。

不能因为参数量只有几千就声称满足实时性。

---

# 十、实验成本：新一轮严格有界

当前旧研究登记预算为12 GPU-wall-hours，
已经基本耗尽。

旧轮次预算不能被静默重置。

请将本轮作为新的明确研究范围进行登记，
保留旧轮次的实际 GPU 用量和结果。

本任务建议的新增加速预算上限：

**4 GPU-wall-hours。**

在新协议中独立登记：

- 新任务范围；
- 预算起点；
- 真实累计 GPU-wall-hours；
- 实际空闲 GPU；
- 存储余量；
- 预期实验队列；
- 每个任务的结果与状态。

这4小时是本轮新任务的最大开销，不是必须用完的配额。

如果现有项目预算守卫还禁止新的 GPU 任务，必须按真实记录创建本轮明确的新预算授权，不得偷偷修改旧任务账本或绕过守卫。若当前运行环境无法确认新任务是否获准，则只执行无新增GPU的代码与监督审计，并报告阻塞项。

旧研究中尚未执行的九次交错计时提案不属于本轮自动执行范围。

不得同时启动该提案消耗本轮算力。

存储：

新增持久数据建议控制在128MiB以内。

训练临时状态尽量保持在内存中。

不得复制 VisionTrack 视频、原始图像或大规模 Stage1 缓存。

必须确保目标盘剩余空间不少于30GiB。

不得删除旧权重、原始研究结果或其他项目数据。

不得停止其他研究任务。

优先使用空闲 GPU。

真实 FPS 测试必须在同一物理设备上进行。

---

# 十一、失败时怎么办？必须按证据转向

以下是本轮允许的全部主要科研转向。

不要额外自由扩展十几个模型。

### 情况 A：Q2 有效监督严重不足

如果在 TRAIN12/13/14 的全部已批准范围中，
无法形成足够的可认证 KEEP/APPLY 事件：

先报告明确的类别、人物和时间依赖支持数。

可以复用已有历史原生事件日志再审计一次，
但不得重建数据集或制造标签。

之后允许做一个极简的 Q1 历史可靠性验证：
检查独立可靠性判断是否能在不学习 Q2 的情况下
提供一个可验证的保守身份保护信号。

这个结果只能称为可靠性诊断，
不能把规则门控伪装成已训练成功的完整JEV。

仍然没有有效信号就结束本轮结构探索，
保留 GMT+MLP。

### 情况 B：Q1 学得好，Q2 没有实际增益

检查：
- Q1 是否真正被 Q2 使用；
- Q2 的可靠性输入是否有有效监督；
- 是否大多数 MLP/GMT 分歧都是 UNKNOWN；
- 是否因动作空间过于保守而总选KEEP。

只允许修复可复现的接口问题，
不能根据 TRAIN16 正负随意修改Loss和标签。

### 情况 C：JEV 的 HOTA 比 MLP 差

如果两个模型监督、证据和预算匹配：
保留普通 MLP 为最佳研究版本。

不强行保留 JEV 作为论文核心。

### 情况 D：HOTA 增加但IDSW/碎片恶化

不宣称身份验证成功。

检查具体身份错误类别，
并保持真实负结果。

### 情况 E：精度与身份证据提升，但速度不合格

仅允许一次不改变模型预测的工程级热点优化。

重点复用 MLP 特征、减少重复计算。

不做大规模网络搜索。

### 情况 F：单序列正向，但没有独立视频

标记：

`DEVELOPMENT_POSITIVE`

而不是：

`SCIENTIFIC_SUCCESS`

冻结候选模型、配置、权重和报告。

不得未经授权读取 DEV17/18/19、
封存20/21/22、官方TEST或Full24。

独立验证仍需要单独的数据资格和合理划分。

### 情况 G：没有任何 JEV 增量

明确写：

`JEV_VERIFIER_INDEPENDENT_VALUE_NOT_ESTABLISHED`

保留原有 MLP 开发结果。

提交完整的负结果与下一步判断，
不无限尝试更多问题数量或更大模型。

---

# 十二、必须回答的 WWW 审稿人问题

最终报告至少回答以下问题。

**R1：为什么原 GMT 需要这个 Verifier？**

必须找出 GMT 现有输出无法充分表达的具体跨摄像头身份证据。

**R2：为什么不是普通 MLP 二次决策？**

必须有输入、监督、参数量匹配的普通验证器。

**R3：为什么不是一个可靠性阈值？**

必须比较置信度、简单历史纯度代理和普通规则。

**R4：JEV 思想究竟体现在何处？**

必须指出两个类型化问题、共享状态、
直接决策和真实因果使用路径。

**R5：两个问题是否都必要？**

必须用去除 Q1、去除 Q2、无类型化问题等消融证明。

**R6：为什么这属于多摄像头跟踪创新？**

必须有跨摄像头历史冲突案例，
而不是普通同摄像头 ReID 分数校准。

**R7：是否真正保持长期 Global ID？**

必须同时看 HOTA、CVIDF1、IDSW、碎片和持续时间。

**R8：为什么速度会下降？**

报告实际热点位置、额外求解成本和完整图像 FPS。

**R9：和 GOT-JEPA 的研究范式有什么相似与不同？**

这里只讨论方法论：
把新模型思想改造成具体跟踪任务所需要的能力。

不要将 GOT-JEPA 的算法结论当作我们方法的实验结果。

**R10：与 Web 有什么关系？**

如果准备投 WWW，研究叙事必须有真实的 Web 任务关联，
例如跨视频身份检索、多摄像头视频实体关联或视频索引。

不能只靠在摘要中添加“Web-scale”字样，
就声称符合 WWW 研究主题。

没有对应任务和实验支撑时，必须如实说明投稿定位风险。

---

# 十三、最终科研验收条件

新的 Final Goal 不能只看有没有正数。

至少分别登记：

`GMT_BASELINE_PRESERVED`

`MLP_POSITIVE_BASELINE_PRESERVED`

`Q1_SUPERVISION_ELIGIBLE`

`Q2_SUPERVISION_ELIGIBLE`

`TYPED_DECISION_CONNECTED`

`NATIVE_ACTION_PARITY`

`DEVELOPMENT_HOTA_GAIN`

`IDENTITY_STABILITY`

`ORDINARY_VERIFIER_COMPARISON`

`JEV_STRUCTURE_ABLATION`

`FULL_IMAGE_FPS`

`INDEPENDENT_VALIDATION`

`SCIENTIFIC_SUCCESS`

**科学成功**必须同时具备：

1. 完整 GMT 真实保留。
2. 同条件超过 GMT。
3. 相对冻结 MLP 有可复现的额外价值。
4. 普通 Verifier 无法解释全部收益。
5. 两问题结构确有必要性证据。
6. 身份一致性改善而不是仅减少切换计数。
7. 实际推理成本合理。
8. 有独立视频和必要的随机种子复现。

缺少任何关键科学证据，都不能标记 COMPLETE。

当前不能因算法在 TRAIN16 单种子正向而称论文已成功。

---

# 十四、GitHub 与实验记录

新建：

`docs/GMT_JEV_VERIFIER/`

`reports/GMT_JEV_VERIFIER/`

至少提交：

- FINAL_GOAL.md
- PREREGISTRATION.md
- BASELINE_FREEZE.md
- DECISION_SUPERVISION_AUDIT.md
- ARCHITECTURE.md
- CONTRACT_TESTS.md
- TRAINING_RESULTS.md
- ONLINE_EVALUATION.md
- EFFICIENCY.md
- QUESTION_ABLATION.md
- REVIEWER_RISK_ANALYSIS.md
- FINAL_RESEARCH_REPORT_ZH.md

以及机器可读结果：

- GOAL.json
- EXPERIMENT_REGISTRY.json
- SUPERVISION_ELIGIBILITY.json
- TRAINING_METRICS.json
- ONLINE_METRICS.json
- SPEED_METRICS.json
- GO_NO_GO.json

每完成一个重要阶段：

1. 写入实际证据，不伪造尚未运行的结果。
2. 保存 checkpoint SHA、配置、随机种子和数据使用范围。
3. 更新 Master Goal 的子状态。
4. Git commit。
5. Git push。
6. 核对远程 HEAD。
7. 在预算与科研资格允许的前提下继续下一阶段。

旧实验不可覆盖。

大模型、原始数据、图像和完整训练缓存不得上传 GitHub。

最后必须提供准确的 GitHub commit 和报告链接。

---

# 十五、最后返回给用户的中文报告格式

请按以下顺序直接回答：

A. 最新 GitHub HEAD、分支和新 Master Goal 状态。

B. Original GMT 和原 MLP 是否完整保留，是否通过等价测试。

C. 新 Q1、Q2 分别使用什么输入、预测什么输出、怎样影响最终匹配。

D. Q1、Q2 的真实监督有多少？是否存在足够独立的纠错样本？

E. 与原 GMT 相比，HOTA、AssA、CVIDF1、IDSW、Frag 分别提升多少？

F. 与现有 MLP 相比，究竟新增了什么收益？

G. 与普通验证器相比，有没有 JEV 式结构独立贡献？

H. 真实完整图像 FPS 变化多少？

I. 哪些结果是开发集结果，哪些通过了独立验证？

J. 最终是否适合 WWW 投稿？当前最缺什么？

最后明确选择：

`CONTINUE_TYPED_JEV`

或

`KEEP_GMT_MLP`

或

`BLOCKED_INSUFFICIENT_SUPERVISION`

并说明依据。

不要用“任务执行完成”代替科学结论。

---

# 十六、现在开始执行

立即执行以下流程：

检查最新 HEAD
→ 创建新研究分支
→ 更新 Final Goal
→ 冻结完整 GMT 与正向 MLP
→ P0 进行同状态方案分歧与可靠监督审计
→ 若监督合格，实现双问题 Typed Verifier
→ 测试 Q1/Q2 是否真实影响 native commit
→ 训练同预算普通验证器与 JEV 验证器
→ 在合法开发范围内完成真实在线比较
→ 满足质量门槛才进行必要消融和完整 FPS
→ 更新研究 Goal
→ 推送全部代码与结果
→ 输出中文结论。

绝对不要再重复：

用新 JEV 整体替换 GTA；

仅通过辅助 Head 分类准确率声称身份关联成功；

只降低 IDSW 却增加身份碎片；

在 TRAIN16 上反复选模型；

把普通 MLP 的提升包装成 JEV 提升；

为了凑创新点增加很多 Question；

为了完成任务而无限扩大 GPU 预算；

为了得到正数而改变既有评测标准。

本次研究的中心问题是：

**在原始 GMT 已经有较强关联能力、普通 MLP 已经取得正向结果的基础上，JEV 式共享历史与类型化身份验证，是否能让模型更可靠地选择“保留原关联”或“采用合法修正”，从而获得额外的全局身份一致性收益？**

如果答案是肯定的，要提供清楚的结构与因果证据。

如果答案是否定的，要承认结果并保留当前最强普通方法。

现在开始执行，不要只返回计划。

**FINAL GOAL: Improve complete GMT through demonstrably useful, efficient, question-conditioned cross-camera identity verification — not by replacing GMT, enlarging JEV, or renaming an ordinary MLP.**
