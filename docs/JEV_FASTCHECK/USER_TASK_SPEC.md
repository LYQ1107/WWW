# Codex 科研执行任务：JEV 闭环训练与架构快速验证

## 0. 你的最终目标

项目：LYQ1107/WWW
GitHub：https://github.com/LYQ1107/WWW

已有 Recovery 分支：

`jev/www-jev-recovery-20261012`

最近已知提交：

`2ab3d47e6ddef7aff9161c1e95dada9918cc01b9`

优先检查本地：

`/home/liuyeqiang/WWW_jev_recovery_20261012`

建议新分支：

`jev/closed-loop-fastcheck-20261014`

### FINAL GOAL

**验证闭环训练能否解决当前 JEV 在真实在线多摄像头多目标跟踪中的身份关联退化；如果不能，明确定位架构问题，并进行一次最小化、有依据的 JEV 结构修复。**

最终希望得到：

- JEV 能正确关联不同摄像头中的同一人物。
- JEV 不会因为自己前面的错误关联而不断污染后续 Global ID。
- 在相同证据、监督和训练预算下，JEV 至少能够与强普通控制器竞争。
- 如果 JEV 获得提升，必须证明提升来自数据、架构或两者中的哪一项。
- 模型必须在真实在线原生状态更新中有效。
- 报告参数量、Stage2 延迟和完整 FPS。
- 不允许通过调低基线、修改 GT 或反复挑选 DEV 结果制造成功。

这次不是无限研究任务。

**采用有时间和算力上限的快速验证。优先用已有模型、合法训练数据和已有缓存。不要从零重训 GMT。**

局部实验失败可以进入下一项预注册实验，但不能无限扩展方法。

---

## 1. 先读取已有结果，禁止重复失败路线

优先阅读：

- `docs/JEV_RECOVERY/RECOVERY_FINAL_REVIEW_20261014.md`
- `docs/JEV_RECOVERY/MASTER_COMPLETION_AUDIT_20261014.md`
- `docs/JEV_RECOVERY/NEXT_SCOPE_ACCESS_AUDIT_20261014.md`
- `reports/JEV_RECOVERY/FULL_ONLINE_RESULTS.json`
- `reports/JEV_RECOVERY/QUESTION_TO_ASSOCIATION_DIAGNOSIS.json`
- `reports/JEV_RECOVERY/QUESTION_TO_ASSOCIATION_INTERPRETATION_V2.json`
- `reports/JEV_RECOVERY/R3_JEV_INDIVIDUAL_MECHANISM_REVIEW.json`
- `gtr/modeling/jev_recovery/model.py`
- `gtr/modeling/jev_recovery/native.py`
- `gtr/modeling/jev_recovery/plans.py`
- `gtr/modeling/jev_oneweek/native.py`

必须理解现有结果：

| 方法 | HOTA | IDSW |
|---|---:|---:|
| Phase XIII Strong Fixed | 75.637 | 105.0 |
| R3 Pair Threshold | 73.738 | 257.3 |
| R3 MLP | 69.314 | 813.7 |
| R3 Attention | 71.367 | 978.7 |
| R3 JEV | 69.120 | 1038.0 |

这是历史 DEV17–19 的三种子汇总，不是独立测试。

注意：除了 JEV，其他新控制器也明显退化。

因此不能预先认定问题全部来自 JEV 网络结构。

另外已有证据：

- JEV 同状态净纠错 +107。
- MLP +151。
- Attention +147。
- Threshold +119。
- 1883 个已认证正确 active-ID 行未进入最终方案。
- 将辅助 Head logits 置零，原最终关联评分不变。
- 移除 joint-plan residual 后，同状态最终选择未变。
- 当前仍缺独立数据授权和正式独立验证。

这些必须作为新实验的研究起点。

不得将已有 NO_GO 改写成 PASS。

---

## 2. P0：工程与数据资格检查

立即检查：

1. 最新 GitHub HEAD。
2. 本地 checkpoint 是否存在且 SHA 对应。
3. 当前 GPU 和显存占用。
4. 可用存储空间。
5. 原有后台任务。
6. 当前原生训练和评测脚本能否复用。
7. 已有 VisionTrack TRAIN 的数据来源、授权及允许用途。
8. 新实验是否需要额外数据授权。

特别注意：

此前阻塞的是新增独立数据的合法来源以及长期安全训练资格。

必须区分：

- 已有授权 TRAIN 的普通 MATCH 训练。
- 需要新授权的独立身份安全数据。
- 原协议封存的独立评价数据。

不要把原来针对长期安全纠错的 8 TRAIN / 3 RESERVED 资格要求，未经分析直接当作所有普通关联实验的通用要求；也不能绕过已有的真实授权限制。

如果当前授权不能支持新的训练或原生回放，禁止启动相关数据实验，继续执行不需要数据访问的代码级测试，并输出准确阻塞项。

不得擅自读取 VisionTrack20/21/22、official TEST 或 Full24。

建立：

`docs/JEV_FASTCHECK/PROTOCOL.md`

冻结新任务的训练范围、评价范围、随机种子、训练预算和成功条件。

旧报告、模型和数据一律保留。

---

## 3. P1：首先验证是否存在训练历史分布不匹配

这是当前最高优先级。

### 核心假设 H1

以前训练 JEV 主要使用旧模型形成的身份历史。

但是在线运行时，JEV 会改变：

- 当前关联结果；
- Global ID 的归属；
- Gallery；
- Memory Bank；
- 后续候选身份；
- 后续可见的历史证据。

因此：

`训练时的历史状态 != JEV 自己运行形成的历史状态`

这一点目前是待验证假设，不是已确认的唯一退化原因。

### 具体实验

冻结现有 JEV 权重。

先不修改模型结构。

在有明确授权的 TRAIN 视频上，分别采集：

**A. Teacher-state trajectories**

旧 Fixed/JEV 基座形成的原生历史。

**B. Self-policy trajectories**

当前 JEV 自己执行关联、自己提交、自己更新历史后形成的原生轨迹。

两组采集必须使用相同的外生视觉输入和检测结果。

可以缓存不依赖关联策略的 Stage1 视觉特征。

但是不能把旧策略的未来 Global ID、Gallery、Bank 或候选集合复制给新策略，冒充它的实际轨迹。

记录每个有效时间点：

- frame、camera、真实检测；
- 当前合法 Global ID 候选；
- 当前视觉证据；
- 每个候选的历史；
- 原始关联分数；
- 当前 JEV 选择；
- 最终 Hungarian/原生合法分配；
- native commit；
- 提交后 Gallery 与 Bank 的变化；
- 下一帧状态；
- 离线 GT 身份标注的有效性。

GT 仅用于离线监督与评估。

不得将当前或未来 GT 作为模型输入。

### 必须做的三个检查

**检查 A：身份历史变化**

统计 JEV 自己形成的 Gallery 与 Teacher Gallery 在以下方面的区别：

- 历史身份混杂程度；
- 候选正确身份覆盖率；
- 高相似度错误身份；
- 错误 Global ID 持续时间；
- 跨摄像头错误传播。

**检查 B：首次状态分歧**

定位 JEV 第一次改变 Fixed 决策的位置。

检查改变是否正确，以及其后是否导致更多错误。

不能因为两个运行的 Global ID 数字不同，就直接认为身份错了。需要有可审计的跨运行身份对应关系。

**检查 C：训练样本有效性**

当前合法候选不存在已认证正确身份时，不允许伪造正确答案。

必须区分：

- 正确 active ID 已存在；
- 正确身份未进入最终方案；
- 正确身份仅存在于合法 Bank；
- 可认证正确身份不存在；
- UNKNOWN；
- 多人混合历史。

对连续帧中的同一错误事件进行去重和分组。

不要把一个持续 100 帧的错误直接当作 100 个独立纠错事件。

### P1 输出

- `STATE_DISTRIBUTION_AUDIT.json`
- `SELF_POLICY_ROLLOUT.json`
- `FIRST_DIVERGENCE_REPORT.md`
- `TRAIN_SAMPLE_ELIGIBILITY.json`

如果已有审计足以证明某项事实，直接复用，不重新计算。

---

## 4. P2：快速验证新训练数据是否真的有用

这是本轮最重要的学习实验。

不修改 JEV 架构，只改变训练状态来源。

准备两种训练数据：

**D0：Original**

现有合格的训练状态和身份标签。

**D1：Closed-loop**

D0 加上模型自身轨迹中的合格困难身份事件。

困难事件可包括：

- 外观相似但真实身份不同；
- 跨摄像头身份混淆；
- 当前关联竞争明显；
- 错误 Global ID 历史；
- 原模型已经正确、不应该干预；
- 有合法正确身份但原策略选错；
- 必须保持 UNKNOWN 的证据不足事件。

对真实错误增加训练覆盖，但不要只训练错误关联。

必须保留大量正确匹配保护样本，防止模型为了纠错频繁改变已经正确的身份。

建议采用有界的 DAgger-style 数据聚合：在训练序列中收集当前策略访问的真实状态，用 GT 离线监督当时可执行的合法动作。

这是已有序列决策思想的适配，不得预先宣称算法创新。

### 最小 2×2 实验

| 模型 | 原训练数据 D0 | 闭环训练数据 D1 |
|---|---|---|
| Ordinary Attention | A0 | A1 |
| JEV | J0 | J1 |

强 Fixed 保持冻结。

四组必须具有：

- 相同合法候选；
- 相同有效训练事件池；
- 相同训练步数；
- 相同评测代码；
- 可比较的新增参数规模；
- 相同在线提交约束；
- 相同模型选择规则。

训练输入如果改变，也必须确保所有模型获得同样的信息。

### 推荐快速预算

优先复用已有权重。

先做：

1. 一段短 TRAIN 视频的真实原生 rollout。
2. 数据完整性和身份标签检查。
3. 每种方案最多 300–500 step 的 smoke/pilot。
4. 只给有实际训练信号的方案升级到 1000 step。
5. 初步验证使用同一个冻结种子与固定训练划分。

不要一开始就做三种子 20k。

初步成功后再追加种子。

在进入实验前把上述预算和升级标准写入新预注册。

### 必须回答

Q1：换成自身历史训练，JEV 是否真的提高当前关联正确性？

Q2：是否减少当前正确身份被反改错？

Q3：局部纠错是否转化为完整在线 HOTA/AssA 提升？

Q4：这种提升是否也发生在普通 Attention 上？

Q5：JEV 是否有相对普通 Attention 的额外收益？

不能把 J1 比 J0 高，直接解释为 JEV 架构独有优势。

---

## 5. P3：如果训练方式不足以解决退化，才验证最小架构修复

禁止直接扩大模型。

已有 JEV 的两个问题：

- Cross-Camera Consistency；
- Identity Stability。

当前报告已经证明两个辅助 Head 的输出 logits 并未直接参与最终评分。

另外联合方案残差在已执行的同状态消融中没有改变所选方案。

所以优先研究：辅助信息如何变成真正可执行的 Global ID 决策。

### 架构方案 M0：保持原结构

复用现有 JEV，作为新架构的同数据对照。

### 架构方案 M1：Direct Association JEV

保留：

- Shared Identity State；
- Dynamic Question；
- Legal Identity Options；
- Question-conditioned Candidate Reader。

移除或关闭没有通过验证的冗余路径。

让模型针对当前合法 Global ID 输出直接用于关联的分数。

不要先预测大量辅助问题，再期待隐藏特征自动变成正确关联。

主损失直接监督当前合法身份决策。

辅助 Cross/Identity Stability 只作为可选辅助监督。

明确 UNKNOWN 和 DEFER 的处理语义。

保持原生全局身份唯一性约束。

### 架构方案 M2：普通 Attention/MLP 等容量对照

给予 M2 与 M1 相同的：

- 身份历史；
- 视觉输入；
- 合法候选；
- 自身轨迹训练样本；
- 监督损失；
- 训练预算。

重点不是让 JEV 表面上拥有更多 Question，而是验证这种问题条件化结构是否真的提高关联效果。

### 必须加入回归测试

- 残差为零时保持强基座原始预测完全一致。
- 候选重新排列不应改变身份选择语义。
- 非法 Global ID 不得被选择。
- 同一相机内部 Global ID 分配不得冲突。
- UNKNOWN 不得被伪造为已知正确身份。
- 重新提交后必须使用新状态继续跟踪。
- 模型输出确实影响实际 native commit。

特别检查当前动态改写 `forward`、hook、全局求解器替换以及 Gallery 特征缓存是否存在状态过期或恢复不完整问题。

如果发现可复现的工程错误，使用专门回归测试修复。

不能未经验证就把工程风险当作已经确定的退化原因。

### 何时不需要执行 M1？

如果 P2 已充分说明训练数据导致了主要问题，并且同一结构经过闭环训练已有明确收益，优先验证该结果。

不要为了完成任务清单而强行改架构。

---

## 6. P4：完整在线验证是最终判据

先在获准 TRAIN 的未参与当前拟合部分验证。

已有 DEV17/18/19 只能作为历史回归，不得继续用它们反复选择最佳方法，也不得宣称它们是独立测试。

只有协议允许并满足数据资格后，才进入真正独立视频评价。

必须输出：

- HOTA；
- AssA；
- IDF1；
- CVIDF1；
- CVMA；
- IDSW；
- MOTA；
- 错误 Global ID 持续时间；
- 错误跨摄像头合并；
- 正确身份保持；
- 已认证 N01/N10；
- UNKNOWN 改配比例；
- candidate coverage；
- native end-to-end FPS。

同时区分：

**局部正确关联**

与

**真实连续跟踪改善**

必须以完整在线冷启动结果作为最后科学判断的核心证据。

---

## 7. 最终成功标准

### 科学成功

在新的合格验证范围中：

1. JEV 的 HOTA、AssA 等主要身份指标相对同条件强基线具有可复现的正向结果。
2. IDSW 和错误身份持续时间没有不可接受的退化。
3. JEV 相对同证据普通 Attention、MLP、Threshold 的独立贡献得到证实。
4. 增量计算成本合理。
5. 不依赖官方 TEST 或反复使用的 DEV 选模。

历史 Strong Fixed HOTA 75.637、IDSW 105 是必须面对的参考成绩，不是不同数据或设置下可以直接比较的统一门槛。

建议把主要改善目标预注册为：争取至少 +0.5 HOTA，同时确保主要身份稳定指标不明显下降。

这只是目标，不是已取得的结果，也不能替代跨视频和跨种子验证。

### 部分成功

若闭环训练明显提高 JEV，但仍然低于强 Fixed：

结论为：

`CLOSED_LOOP_TRAINING_HELPFUL_BUT_NOT_SOTA`

继续完成预先允许的必要分析，不能宣布 JEV 突破。

若 JEV 与 Attention 同样受益：

结论为：

`TRAINING_GAIN_NOT_JEV_SPECIFIC`

如果改架构后 JEV 仍没有优势：

结论为：

`JEV_INDEPENDENT_ADVANTAGE_NOT_ESTABLISHED`

必须承认普通方法可能更适合当前任务。

不得为了得到正数反复更换评价集、阈值或随机种子。

---

## 8. 强制回答五个审稿人问题

所有报告必须回答：

**问题一：为什么不是普通二次关联？**

必须比较真正相同候选、相同求解器预算下的二次关联控制。

**问题二：为什么不是固定阈值或可学习阈值？**

必须比较强 Pair Threshold，而不是人为设置弱阈值。

**问题三：为什么不是普通 MLP / Attention？**

对照必须获得相同信息与闭环训练样本。

**问题四：参数量与 FPS 代价是多少？**

实际实例化统计：

- 原始系统总参数；
- 净增减参数；
- 可训练参数；
- Stage2 p50/p95；
- 完整图像场景 FPS；
- 新方法相对强基线的 FPS 变化。

不把不同 GPU 的历史 FPS 当作直接速度对照。

**问题五：是否真正体现 JEV 思想？**

必须验证共享状态、类型化问题、动态合法选项和直接决策读出是否发挥作用。

如果动态 Question 与普通 Attention 没有区别，不能仅靠命名声称结构创新。

---

## 9. 时间和资源控制

用户时间紧张。

请按 FAST VALIDATION 执行，不要变成长期开放式探索。

建议预算：

- P0/P1：以已有日志和数据为优先，避免重新跑全量。
- P2：先短序列、单种子、最多 1000-step Pilot。
- P3：只有 P2 暴露明确结构瓶颈时执行。
- P4：仅最优合格方法升级完整验证。

建议第一轮验证的新增 GPU 预算上限为 12 GPU-hours，并以尽可能少的墙钟时间执行。实际配置及并行方案按机器状态制定，不得抢占现有任务。

若机器具备 4×A100 40G，可以并行不同方法，但应先验证相同输入、环境和资源可用性。

发现显存或磁盘不足时优先复用已有缓存、分块写入。

禁止重复生成完整视觉缓存。

严禁删除旧 checkpoint 和历史原始结果。

空间不足时，先清理本轮可再生的临时数据，保留 manifest 与原始研究证据。

若正式数据授权不满足，绝不能为追求快速结果绕过访问门槛。

普通工程错误允许自动修复、重跑相关小测试。

某一科学假设不通过，可以推进另一项已预注册的合法实验。

**不允许无限增加模型版本，不允许未记录的临时超预算训练。**

---

## 10. GitHub 和报告要求

建立：

`docs/JEV_FASTCHECK/`

以及：

`reports/JEV_FASTCHECK/`

至少生成：

- `PROTOCOL.md`
- `BASELINE_AND_ACCESS_AUDIT.md`
- `STATE_DISTRIBUTION_REPORT.md`
- `CLOSED_LOOP_TRAINING_RESULTS.md`
- `ARCHITECTURE_DIAGNOSIS.md`
- `ONLINE_EVALUATION.md`
- `REVIEWER_FIVE_QUESTIONS.md`
- `FINAL_DECISION.md`
- `EXPERIMENT_REGISTRY.json`
- `FINAL_RESULTS.json`

每完成一个主要阶段：

1. 保存真实指标。
2. 保存训练配置、随机种子、数据来源和 checkpoint SHA。
3. 更新当前阶段结论。
4. Git commit。
5. 推送当前研究分支。
6. 核对远程 HEAD。
7. 继续下一项已经批准、仍然有必要的实验。

不要上传大型图像、数据集、完整视频、模型权重和巨大缓存。

最终向我报告：

**A. 换训练数据到底有没有用？**

**B. 造成退化的主要证据是什么？**

**C. 现有 JEV 是否值得保留？**

**D. 是否真的需要重构 JEV？如果需要，哪一处必须改？**

**E. 最好的真实 HOTA、AssA、CVIDF1、IDSW 与 FPS 是多少？**

**F. 是否比 Strong Fixed、Threshold、MLP、Attention 更好？**

**G. 哪些结果仍缺独立验证？**

**H. 下一步应该继续 JEV，还是转向已经被证据支持的普通关联方法？**

不要把“所有脚本都已执行”当作研究成功。

## 11. 立即执行

现在开始：

**先确认授权和现有工程状态 → 复用历史结果 → 快速采集 JEV 自身跟踪状态 → 训练原结构的闭环版本 → 与普通 Attention 公平比较。**

如果闭环训练没有解决退化，再按已冻结的条件进行最小 JEV 结构修复。

优先获得可信的实验结论，不要为了复杂而复杂。

本轮的最终目标是：

**搞清楚我们当前 JEV 究竟是训练方式不对，还是决策架构不合适，并尽可能得到一个真实、稳定、能够改善在线多摄像头身份跟踪的模型。**

START NOW.
