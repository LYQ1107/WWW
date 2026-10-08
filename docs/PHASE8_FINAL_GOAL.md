# FINAL GOAL — WWW / JEV PHASE VIII
## Corrective Candidate Association Learning
### 从真实纠错监督到候选条件化身份决策

**项目仓库：** https://github.com/LYQ1107/WWW

**任务级别：** 完整科研阶段任务，允许自主执行代码、实验和验证，但必须遵守科学 Gate 与存储保护规则。

**研究主线：** Online Multi-Camera Multi-Target Tracking

**底座：** GMT / VisionTrack

**目标会议：** The Web Conference（WWW）

---

# 0. FINAL GOAL — 最终研究目标

本次 Phase VIII 的最终目标是：

**建立一个基于真实在线因果数据的 Candidate-Conditioned JEV 身份关联模型，系统验证数据规模、训练轮数、模型容量、监督目标、样本组织和网络结构对关联决策的影响，证明或证伪 JEV 是否能在相同感知输入、计算预算和原生关联求解器下，超过固定规则、可学习阈值、普通 MLP 和候选集合网络。**

最终必须回答六个研究问题：

1. 当前 JEV 的失败是否因为有效训练样本太少？
2. 是否因为训练数据中 ACCEPT 占据绝对多数，缺少真正的纠错动作？
3. 是否因为训练轮数不足、模型没有充分收敛？
4. 是否因为模型参数量和表示容量不足？
5. 是否因为当前 JEV 的 state-only 动作评分结构无法充分表达动态候选身份关系？
6. 是否因为当前反事实 utility 及其样本权重不能正确反映真实身份纠错收益？

此外，需要保持长期研究方向：

- MATCH：学习正确的身份关联。
- MEMORY：维护可靠的身份记忆。
- REACTIVATION：恢复长期消失的身份。

这三个模块最终希望构成一个统一的 Identity Lifecycle Decision Engine。

但是 Phase VIII 的第一优先级是 MATCH。只有确认 MATCH 的有效监督与候选级方法后，才开展后续两模块的条件研究。

**本次任务不以必须得到正向结果为要求，而以得到可信、可归因、可复现的研究结论为要求。**

---

# 1. 已有结果与保护锚点

开始前，先核对 GitHub 分支及服务器文件。不能假设所有本地运行状态都与远程完全一致。

Phase VII branch：

`jev/www-jev-phase7-causal-structured-20261008`

已核验的当前 HEAD：

`ecc0e63f544cbbc272797c17ea285b0b8cce5f80`

Phase VI HEAD：

`40cbc0ecc22bc16c4e3602eefe5eb06e2d6e319e`

Phase V HEAD：

`a37083dac0a23eb3846cb252eeb975982aaaabc9`

永久保护 B2 checkpoint SHA256：

`f2aa3dd2b564d90bfbfb62dc0518b0b2107a931ed7134f8c0d19f5d152b94ed7`

已知 B2 原始位置：

`/home/liuyeqiang/WWW_jev_phase5_runtime/20261007/minimal_training/B2/calibration/model_calibrated.pth`

Phase VII 原始实验归档：

`/home/liuyeqiang/WWW_jev_phase7_runtime/20261008_p05_v1`

必须阅读：

- `docs/PHASE7_FINAL_RESEARCH_REPORT.md`
- `docs/PHASE7_PUBLICATION_POLICY.md`
- `docs/PHASE7_ARCHITECTURE_DESIGN.md`
- `reports/JEV_PHASE7/FINAL_GO_NO_GO.json`
- `reports/JEV_PHASE7/LEARNABILITY_AUDIT.json`
- `reports/JEV_PHASE7/NATIVE_CAUSAL_MINISET_V1_AUDIT.json`
- `reports/JEV_PHASE7/PHASE7_REASSOCIATION_ABLATION.json`

Phase VII 已知：

- 69 次完整视频对照已经完成。
- 三个 controller-heldout 为 video09、10、11。
- GMT HOTA 84.6234、AssA 83.7393。
- Fixed Rule HOTA 85.0300、AssA 84.5577。
- B2 HOTA 84.9749、AssA 84.4476。
- Fixed Rule 的 pooled 表现高于 B2。
- Train07 15 条有效 MiniSet 监督全部为 ACCEPT。
- Val06 11 条有效监督全部为 ACCEPT。
- 完整 train07 事件流发现 28 个疑似可纠错提议事件；val06 为 0，但这些候选机会还需要检验全局合法性和真实 native 改配收益。
- MEMORY 单次 WRITE/SKIP 在 H64 内仍无有效 identity utility 差异。
- relative REACTIVATION production hook 仍未建立完整契约。
- Candidate-Conditioned JEV 尚未真正训练。
- Unified JEV 尚未真正训练。

不得重新将上述结论描述为 GO。

---

# 2. P-1：存储空间审计与安全清理

**这一步必须最先执行。**

用户明确授权清理确认完全无用的 checkpoint 与废弃中间文件，以释放空间。

但授权仅限于已证明不再需要、也没有复现用途的文件。

不能因为一个实验 NO-GO，就认定它的 checkpoint 不再需要。

## 2.1 先调查真实存储情况

执行：

- `df -hT`
- `df -i`
- 对项目所在文件系统执行受限的 `du -xhd1`
- 检查当前运行中的 Python/Codex/GMT 任务
- 检查正在进行的训练、评估、归档和下载任务
- 确定数据集、工作树、checkpoint、日志、缓存和归档的实际位置

重点调查：

- `/home/liuyeqiang`
- `/data1/liuyeqiang`
- 当前 WWW 项目工作树
- Phase V、VI、VII runtime
- 其他实际被项目配置引用的存储目录

不要假设这些目录都在同一个文件系统。

禁止全盘无边界递归扫描、删除系统缓存或影响其他用户的任务。

输出：

`reports/JEV_PHASE8/STORAGE_BEFORE.json`

至少包含：

- 每个文件系统总容量、已用、可用
- 项目目录占用
- checkpoint 总容量
- 原始实验归档容量
- 临时文件容量
- 数据集容量
- Git 仓库对象容量
- 当前活跃任务和相关路径

## 2.2 建立 checkpoint 依赖清单

枚举项目相关目录中的：

- `.pth`
- `.pt`
- `.ckpt`
- `.safetensors`
- optimizer / scheduler state
- EMA checkpoint
- 临时 training snapshots

对每一个 checkpoint 记录：

- 完整绝对路径
- 文件大小
- SHA256
- 修改时间
- 原实验 run ID
- 来源 commit
- 是否被代码或配置引用
- 是否被 JSON/Markdown/report manifest 引用
- 是否被运行中的进程使用
- 是否用于复现实验结果
- 是否为目前唯一一份有效副本
- 删除它是否破坏科研审计

输出：

`reports/JEV_PHASE8/CHECKPOINT_INVENTORY.csv`

## 2.3 checkpoint 分类

所有文件分为以下四类：

**A — PROTECTED**

永久保留。

包括：

- GMT backbone / `model_20000.pth` 及当前正式依赖的权重
- Phase V B2
- Phase VI 已归档的核心模型
- Phase VII 研究报告和消融实验依赖的 checkpoint
- 任何基线结果、失败实验、论文表格、图表需要复现的模型
- 当前正在使用或未来明确列入研究协议的 checkpoint

**B — REPRODUCIBILITY_REQUIRED**

当前可能不运行，但仍有科研复现价值。保留。

**C — VERIFIED_ORPHAN**

同时满足全部条件才允许删除：

1. 无活跃任务使用。
2. 不被当前或未来已登记任务引用。
3. 不被训练恢复配置引用。
4. 不被任何保留报告、预测、评估或科研归档依赖。
5. 不属于需要保留的负实验模型。
6. 不是某个有效模型的唯一副本。
7. 不是必要的模型初始化或数据处理依赖。
8. 已记录路径、SHA、大小、来源及删除理由。
9. 已进行第二次引用与进程检查。
10. 删除不会破坏现有实验重放或结果核验。

符合上述条件，可以直接按显式路径清理。

**D — UNKNOWN**

无法证明无用的全部保留。

## 2.4 执行删除

先生成：

`reports/JEV_PHASE8/SAFE_DELETE_MANIFEST.json`

`reports/JEV_PHASE8/DELETE_DRY_RUN.md`

审核程序必须支持：

- 默认 dry-run
- 基于精确路径执行
- 删除前再次核验 SHA256 和依赖
- 发现路径变化或正在被使用就拒绝
- 输出逐文件执行结果和释放空间

禁止使用无边界 `rm -rf` 或根据通配符批量删除未知 checkpoint。

不得删除 Phase V/VI/VII 已发表科研证据，也不得重写 Git 历史来强行释放空间。

此外，可以审查明显废弃的 `.tmp`、崩溃中间目录和重复 staging 产物，但必须沿用相同依赖检查。

清理完成后输出：

`reports/JEV_PHASE8/STORAGE_AFTER.json`

`reports/JEV_PHASE8/STORAGE_CLEANUP_REPORT.md`

明确：

- 删除哪些文件
- 为什么确认完全无用
- 保留哪些大型文件
- 实际释放多少空间
- 是否仍存在空间压力

若找不到可以安全删除的 checkpoint，报告 0 GB 并继续研究，不得为了完成清理指标而删除不确定文件。

## 2.5 新阶段的存储规范

从 Phase VIII 开始：

- 新训练优先保留必要的 best / last checkpoint，不默认保存每一轮。
- 不重复缓存完全相同的 GMT perception。
- 原始 immutable dataset 和 perception cache 优先复用。
- 原生反事实分支采用有界数量和可恢复 manifest。
- 大型中间产物按照生命周期管理。
- 每个阶段结束时进行空间审计。
- 不因 CPU/GPU 空闲而预先产生百万条数据。
- 严格遵守 Phase VII 的 compact review publication policy。

GitHub 默认只提交：

- 源码
- 文档
- 训练及评价配置
- 分割与随机种子
- 汇总及逐视频指标
- 小规模诊断实例
- 图表
- SHA/provenance manifests
- 必要的可运行测试

完整大型原始证据保存在经过验证的服务器归档中。

不得默认推送几 GB 的原始 predictions、state snapshots、journal、fork logs、perception arrays 或临时 checkpoint。

每一个大文件上传前必须有明确科研复现必要性。

---

# 3. Phase 0：冻结研究范围

新建独立分支：

`jev/www-jev-phase8-corrective-association-20261008`

如果该分支已存在，则检查实际状态，使用不会覆盖其他实验的独立 worktree。

冻结：

- GMT backbone checkpoint
- detector/perception outputs
- native association semantics
- evaluation protocol
- GT coordinate contract
- Phase V/VI/VII 保护锚点

Phase VIII 禁止：

- 改写已有原始 B2 checkpoint
- 用 legacy trace 代替 native counterfactual truth
- 使用官方 TEST 选模型
- 自动启动 Full24 H8
- 为了获得正向结果修改已有评估指标
- 把旧 research resolver 当作 production parity
- 在没有实际监督的情况下训练 Unified JEV

所有新任务必须记录明确的 start commit、source SHA、配置和独立输出目录。

---

# 4. Phase A：寻找真正的可纠错 MATCH 事件

## 4.1 核心目的

验证真实 GMT 数据中究竟存在多少：

**当前身份关联错误，但正确身份已出现在合法候选集合中，并且模型能够通过正确选择改善后续在线关联的事件。**

不是简单统计有多少检测匹配错误。

必须区分：

1. GMT 提议正确。
2. GMT 提议错误，但正确身份根本不在候选集合。
3. 正确身份存在，但受到全局匹配约束无法合法提交。
4. 正确身份存在，可以合法重新分配。
5. 重新分配后确实产生更好的当前身份结果。
6. 重新分配在后续 H8/H16/H32 中持续带来收益。

正确候选是否存在，不能只比较 predicted track ID 和 GT ID 的整数值。

必须通过可靠的历史身份锚点与 GT 对应关系审计。

同一 GT 身份可能对应多个已有 predicted IDs，必须记录歧义，不能把编号偶然相同当成正确。

## 4.2 数据来源

优先使用 VisionTrack 官方 TRAIN 数据。

Phase V–VII 已反复使用 video01、02、03、05、06、07、09、10、11、23、24，video04 也曾用于调试。

其他 TRAIN 视频可作为候选来源，但必须先检查历史使用记录。

在读取新视频的纠错结果前，预先冻结：

- 训练视频列表
- 验证视频列表
- controller-heldout 列表
- 固定种子
- 场景和视频划分规则
- 事件筛选规则

不得通过查看最终 HOTA 或候选模型表现，重新选择有利的验证视频。

不得把已用于结果解释的旧视频称为全新独立测试。

## 4.3 轻量级全量机会扫描

先只运行必要的 native proposal/candidate audit。

不要对所有事件展开 H32 反事实。

记录每个 MATCH event：

- video/frame/view/event key
- GMT 原始 proposal
- candidate track IDs
- candidate scores
- 原始关联正确与否
- GT 是否可判定
- 候选集合是否包含正确身份
- 候选排序
- candidate rank / margin
- 全局分配冲突
- 轨迹年龄和记忆质量
- 首次身份关联还是长期重关联
- 是否可以合法直接提交正确候选
- 是否需要改变其他行的身份分配

输出总体统计与困难事件的稀疏索引。

禁止将全部当前检测乘全部历史 ID 的巨大矩阵无条件长期保存。

## 4.4 Candidate oracle 诊断

计算不同层次的上界：

**Oracle-1：Candidate Availability**

正确身份是否出现在当前可用集合中。

**Oracle-2：Feasible Assignment**

在一对一和跨视角约束下，是否存在全局合法的正确分配。

**Oracle-3：Native Corrective Action**

原生在线状态允许的动作，能否真正提交该分配。

**Oracle-4：Causal Future Benefit**

执行纠错后是否在预先定义的未来窗口中改善身份连续性或长期效用。

这些 oracle 只能用于离线诊断，不能作为运行时 GT-aware 控制器。

## 4.5 Phase A 的通过要求

需要获得真实可纠错事件，而不仅仅是大量普通 ACCEPT。

阶段目标建议为：

- TRAIN 至少约 50 个经核验的真实可纠错机会。
- Validation 至少约 20 个来自独立视频的真实可纠错机会。
- 纠错来源包含多种关联状态，而非大量重复的同一个错误窗口。
- 至少存在两个以上具有实质不同最优身份选择的动作或候选结果。
- 这些机会确实能够在 native runtime 中合法执行。

这些数量是建议的研究启动门槛，不是声称已知的数据集存在这么多机会。

必须额外报告视频级有效样本数和 effective sample size；高度相关的连续帧不能当成独立样本。

如果真实数据无法满足，允许输出 `BLOCKED_INSUFFICIENT_CORRECTIVE_OPPORTUNITY`。

不得为了通过门槛伪造错误、增加假的旧身份或偷换验证视频。

输出：

`reports/JEV_PHASE8/CORRECTIVE_OPPORTUNITY_AUDIT.json`

`docs/PHASE8_CORRECTIVE_OPPORTUNITY_REPORT.md`

---

# 5. Phase B：构建 Native Corrective Association Dataset v2

只对 Phase A 确认的有效事件进行有界因果数据构建。

## 5.1 数据集目标

数据集必须能够回答：

- 当前检测到底属于哪个候选身份？
- 哪个候选身份更可靠？
- 选择该身份后，其他目标会不会受到影响？
- 如果选择 START_NEW，会发生什么？
- 当前纠错是否改善未来轨迹？
- 哪个动作带来的未来代价最小？

这是决策训练数据集，不是新建原始视频数据集。

## 5.2 样本组织

明确保留三种数据：

**Natural Distribution**

完整真实事件分布，用于评价真实决策频率与最终 HOTA。

**Hard Corrective Events**

原提议错误且存在可行正确候选的事件。

**Hard Negative / Regression Events**

原提议本来正确，但某个具有迷惑性的替代候选容易导致反改错。

训练时可以对困难事件进行分层采样。

验证阶段必须保留自然分布评价，并另外报告事先定义的困难事件子集。

不能把 1:1:1 的人为动作平衡分布当作真实 online 事件分布。

## 5.3 真实反事实标注

从完全相同的 native mutable state 出发：

- CONTROL
- 保留原始关联
- 选择某个合法替代候选
- START_NEW
- 在可行时进行有界联合重分配

比较 H8、H16、H32 的未来结果。

保存：

- 完整状态 fingerprint
- 起始 score/candidate 矩阵
- 实际提交 ID
- 每个动作后立即改变的状态
- RNG provenance
- 未来 identity correctness
- 错误身份持续时间
- false merge / false birth
- wrong-write diagnostics
- utility components
- censor/tie/unknown flags

所有分支使用一致的外生视频输入和固定的后续策略定义。

禁止通过缓存未来 GT 决定运行时动作。

对于有行间冲突的 Hungarian 关联，不能默认每条独立匹配边的 Q 值可加。必须进行有限的 joint conflict 反事实测试。

## 5.4 监督目标分成两个层次

第一层：

**Candidate Correctness Supervision**

学习当前候选是否代表正确的历史身份。

第二层：

**Counterfactual Long-Term Utility**

学习合法决策对未来跟踪结果的影响。

不能把两者混为同一个概率。

检查现有新建身份惩罚、身份保持奖励等 utility 项是否造成系统性 ACCEPT 偏好。

比较合理的 utility 组成与权重敏感性，但所有候选权重和选择规则必须在验证结果出现前冻结。

输出：

`NATIVE_CORRECTIVE_V2_MANIFEST.json`

`NATIVE_CORRECTIVE_V2_AUDIT.json`

`docs/PHASE8_DATASET_CONTRACT.md`

以及可运行的加载器、测试和小规模数据。

---

# 6. Phase C：六大失败原因诊断

此阶段必须先测试最小实现和训练可学习性，不立即构建一个巨大的新模型。

## C1：Tiny-Set Overfit Sanity

使用经过审计的小规模训练样本，检查：

- 当前 State MLP
- 当前 State JEV
- Candidate MLP
- 简单 Candidate DeepSets

能否拟合可靠的候选正确性标签。

分别报告 train loss、train accuracy、candidate rank 和动作分布。

如果所有模型都无法拟合，应优先排查输入可分性、身份标签契约、归一化、mask 或冲突标签。

不能直接归因于参数容量不足。

## C2：数据量曲线

固定网络结构和训练预算，对不同训练数据规模进行实验。

建议观察：

- 25%
- 50%
- 75%
- 100%

这里按独立纠错事件或事件组划分，而不是随机抽取高度重复的帧。

观察：

- Train/Val 性能
- Corrective Recall
- Wrong Correction Rate
- Candidate MRR / Rank-1
- Native closed-loop AssA/HOTA

如果随真实有效样本增加而稳定提高，支持数据量不足假设。

## C3：训练次数曲线

固定训练集与网络参数，比较：

- 5 epochs
- 20 epochs
- 50 epochs
- 100 epochs

记录完整训练曲线、验证曲线、梯度和 checkpoint。

不必对每个 epoch 保留模型文件，只保留必要快照及最佳/最终 checkpoint。

测试是否出现：

- 明显欠拟合
- 已经收敛
- 后期过拟合
- 概率校准恶化
- 在线行为分布漂移

如果训练早已收敛，不允许简单把 epoch 提到更高来掩盖问题。

## C4：容量曲线

建议比较近似：

- 8K
- 34K
- 128K
- 500K 参数

具体实现以实际参数量为准。

必须给所有模型一致的数据和监督目标。

统计：

- Parameters
- FLOPs/MACs
- Training/validation performance
- Online latency
- Memory usage

判断更大容量是否改善真正的候选区分能力，而不是仅提高训练集拟合。

## C5：数据组织和损失函数

在相同有效数据上比较：

- Natural Sampling
- Stratified Hard Sampling
- Correctness CE
- Focal Loss
- Candidate Ranking Loss
- Utility Regression / Advantage Ranking
- Correctness + Utility Multi-task

不能为了平衡而改变验证集自然分布。

所有数据增强必须明确标记真实 native 事件和人为模拟扰动。

## C6：输入归一化

必须实现并冻结训练数据统计得到的 normalization。

训练和 native runtime 使用完全相同转换。

比较：

- Raw input
- Standardized input
- LayerNorm controls

所有 MLP、JEV、DeepSets baseline 都必须获得公平的输入处理。

特别注意 Phase VII 动态阈值和未归一化 MLP 的崩溃记录，不能因为某组实现数值不稳定就宣布整个方法类别无效。

输出：

`reports/JEV_PHASE8/LEARNING_CURVES.json`

`reports/JEV_PHASE8/DATA_SCALING.json`

`reports/JEV_PHASE8/CAPACITY_ABLATION.json`

`reports/JEV_PHASE8/SAMPLING_LOSS_ABLATION.json`

`docs/PHASE8_FAILURE_CAUSE_ANALYSIS.md`

每一项必须给出证据、结论和无法确定的部分。

---

# 7. Phase D：Candidate-Conditioned JEV 新架构

Phase C 证明候选关联监督有效以后，正式进入架构开发。

## D1：设计原则

不再只对压缩的 state 预测：

`ACCEPT / REASSOCIATE / START_NEW`

而应当从真实候选集合中评价：

- Detection X → Track A
- Detection X → Track B
- Detection X → Track C
- Detection X → NEW

每个候选使用共享 scorer，不以绝对轨迹 ID 为固定分类类别。

候选顺序变化不应该改变候选自身的输出语义。

## D2：输入

优先复用 Phase VII 已设计的：

- 64D state
- 实际 12 维候选 online evidence
- 合法候选 mask
- GMT raw scores
- 候选竞争与关系上下文
- 历史质量与真实 memory fields
- 当前 question type

Candidate IDs 只能作为关联引用与索引元数据，不能直接作为模型可记忆的身份整数特征。

不得臆造 motion、depth、uncertainty 或其他目前不存在的特征。

## D3：候选编码与动态动作选择

首先实现轻量 DeepSets / candidate scorer：

- shared candidate encoder
- permutation-invariant set context
- candidate-relative scoring
- semantic NEW option
- legal masking

然后开发 Candidate-Conditioned JEV：

- shared state/question representation
- candidate-specific action values
- candidate competition
- predicted correctness
- predicted long-term utility

初版目标参数量接近 B2 的 34K。

参考 Phase VII 设计文件中的 32,093 参数原型和 33,887 参数匹配版本。

实际实现时重新测量准确参数量与延迟。

## D4：原生关联提交接口

当前 GMT production hook 还不能保证直接接收某个候选的价值并提交其身份。

必须先设计独立、可测试的候选分配接口：

1. 取得真实 score matrix 和合法候选。
2. 接收模型提供的候选评分。
3. 施加全局一对一及跨摄像头身份约束。
4. 产生合法匹配。
5. 使用真实 native commit。
6. 维护相同的 MEMORY、stale-bank、birth/lost 状态语义。
7. 将实际提交结果反馈到下一帧。

改变 solver score 或 cost definition 时，MLP、DeepSets 和 JEV 必须使用相同的提交算子。

禁止仅让 JEV 使用更强的 Hungarian 或额外候选，而 baseline 仍使用旧求解器。

必须增加以下测试：

- Candidate permutation equivariance
- Duplicate candidate suppression
- Legal candidate masking
- No two detections illegally claim one identity
- Correct NEW semantics
- Native committed-ID parity
- Mutable state fingerprint
- Deterministic RNG
- Candidate missing / no feasible assignment
- No GT or future information in inference

---

# 8. Phase E：最关键的公平模型实验

必须比较：

**B0 — GMT OFF**

原生基线。

**B1 — Fixed Rule**

使用同样合法候选与原生 solver 的固定规则。

**B2 — Dynamic Learnable Threshold**

使用完整当前状态的可学习规则。

**B3 — Strong Candidate MLP**

读取与 JEV 相同的候选证据，参数规模匹配。

**B4 — Candidate DeepSets**

验证候选集合结构本身带来的优势。

**B5 — Existing State-only JEV**

原有结构对照。

**B6 — Frozen Original B2**

永久正向 anchor。

**B7 — Candidate-Conditioned JEV**

新的主要研究对象。

**B8 — Candidate-Conditioned JEV + Causal Utility**

检查加入长期反事实监督的增量价值。

所有模型必须使用：

- 同一个 perception
- 同一候选集合
- 同一 Train/Val split
- 同样的合法候选与动作空间
- 同一 native solver
- 相同训练数据
- 可比的 optimizer/update budget
- 相同 validation-only calibration
- 可比参数和计算量
- 相同的最终 TrackEval 配置

优先执行同参数、同候选、同正确性监督实验。

然后只改变 utility 监督或模型结构进行受控消融。

不要同时改变数据、网络、损失和重关联算法。

若新 Candidate JEV 超过 State-only JEV，却没有超过 Candidate MLP/DeepSets，只能证明候选表示很重要，不能证明 JEV 特有架构优势。

---

# 9. Phase F：独立完整视频在线验证

使用预先冻结且未参与本轮模型选择的 controller-heldout。

所有方法执行真实 mutable-state online closed-loop。

不允许直接使用 GT 修正轨迹，不允许复用某个控制器未来动作作为其他控制器的真实闭环。

主指标：

- HOTA
- AssA
- IDF1
- IDSW
- MOTA
- Frag

机制指标：

- Candidate Top-1 / MRR
- Corrective Recall
- Correction-to-Regression ratio
- Wrong-Identity Duration
- False Merge
- False Birth
- NEW 频率
- Candidate Availability
- REASSOCIATE trigger frequency
- Identity contamination
- Online p50/p95 latency
- Extra GPU memory
- FLOPs / model parameters

必须报告逐视频结果与使用 TrackEval 正确合并得到的 pooled 结果。

不得使用简单 HOTA 算术平均代替官方 pooled。

如果有效样本量允许，至少使用三个预冻结 seed 对主要学习模型重复训练。

以视频或独立场景作为统计重复单位，不能把重叠事件窗口作为独立样本制造显著性。

针对 WWW，保留小规模可验证的跨摄像头身份索引查询实验，报告错误身份连接、查询污染与查询延迟。

不要求通信或带宽仿真。

---

# 10. Phase G：MEMORY 与 REACTIVATION 条件研究

此阶段保留作为整个 Identity Lifecycle 的长期主线，但不允许越过 MATCH 与原生接口 Gate。

## G1 — MEMORY

不继续盲目扩大原 WRITE/SKIP 训练。

首先研究：

- Gallery update policy
- Confidence weighted EMA
- Memory sample replacement
- Prototype reliability
- Diverse historical sample retention
- Actual future bank READ
- Wrong identity write contamination

借鉴 QDTrack、MeMOTR、Deep-OC-SORT 中的可靠性更新思想。

真实比较当前固定规则与更新策略的因果效果。

只有出现足够的可识别行动收益，才训练 MEMORY JEV。

## G2 — REACTIVATION

首先补齐 production relative REACT hook，并检查：

- 合法旧身份候选数量
- Bank threshold 0.4 下的候选覆盖
- 在预冻结阈值压力测试下的 false reactivation
- 正确恢复机会
- START_NEW 与 REACTIVATE_OLD 的真实区别

放宽阈值不能被当成自动改善，必须同时评价错误恢复与误合并。

不能制造假的 stale candidates。

## G3 — Unified

只有 MATCH、MEMORY、REACTIVATION 三个任务分别满足数据和 native parity Gate，才考虑：

- Shared JEV core
- Typed task adapters
- Cross-question transfer
- Joint utility learning
- Independent heads vs shared core
- MATCH-only vs Unified full lifecycle

如果某个模块没有合格监督，保持 BLOCKED，不要为了完整故事训练没有科学依据的共享 head。

---

# 11. 预注册 GO / BLOCKED / NO-GO

在新 heldout 指标出现前，冻结正式研究判断标准。

## 数据 Gate

必须满足：

- 真实且可核验的纠错事件。
- 正确候选和合法替代动作充分覆盖。
- TRAIN 和 Validation 有独立有效机会。
- 真实 native factual replay 与 commit/state parity PASS。
- 没有 GT leakage。
- 不存在仅由人为新建惩罚造成的虚假优势。
- 不将高度重复的困难片段当作独立样本。

## MATCH 架构 Gate

需要满足：

- Candidate JEV 相对于 GMT 和固定规则，在预注册 heldout 上改善 HOTA、AssA。
- 相对于强 Candidate MLP / DeepSets，有额外可归因价值。
- 在困难关联上提高纠错率，不依赖大量反改错或错误合并。
- 保留完整闭环优势，而不只在离线 Rank-1 提升。
- 参数、数据、求解器和训练预算公平。
- 不因单序列正向而宣布泛化成功。

建议沿用此前研究的最低效果目标作为主要实用增益门槛：

- 相对最强预注册普通基线，pooled HOTA 至少 +0.1。
- pooled AssA 至少 +0.2。
- 多个独立视频正向，不能依靠单一序列主导。
- Corrective Recall 提高且 Regression 不抵消真实收益。
- 在线推理开销满足预先冻结的实时约束。

上述阈值是目标，不保证模型一定能通过。统计不确定性必须单独报告。

## Stop conditions

如果没有真实纠错样本：

`BLOCKED_DATA_OPPORTUNITY`

如果真实候选无法合法提交：

`BLOCKED_NATIVE_CANDIDATE_INTERFACE`

如果小样本无法拟合：

`BLOCKED_REPRESENTATION_OR_LABEL_CONTRACT`

如果训练拟合但不泛化：

`NO_GO_GENERALIZATION`

如果 JEV 未超过参数匹配的 MLP/DeepSets：

`NO_GO_JEV_ARCHITECTURE_ADVANTAGE`

如果 MEMORY 无有效监督：

`BLOCKED_MEMORY_IDENTIFIABILITY`

如果 REACT native hook 不完整：

`BLOCKED_REACTIVATION_PARITY`

如果三模块没有联合证据：

`NO_GO_UNIFIED_CLAIM`

本次任务即使 MATCH 获得 GO，也不自动授权 Full24 或官方 TEST。

必须输出正式决策和后续建议，供用户单独审核。

---

# 12. 最终交付物

建立：

`docs/PHASE8_FINAL_GOAL.md`

作为不可随意改写的研究目标。

同时输出：

- `docs/PHASE8_RESEARCH_PLAN.md`
- `docs/PHASE8_CODE_AUDIT.md`
- `docs/PHASE8_CORRECTIVE_OPPORTUNITY_REPORT.md`
- `docs/PHASE8_DATASET_CONTRACT.md`
- `docs/PHASE8_FAILURE_CAUSE_ANALYSIS.md`
- `docs/PHASE8_CANDIDATE_JEV_ARCHITECTURE.md`
- `docs/PHASE8_FAIR_BASELINE_PROTOCOL.md`
- `docs/PHASE8_FINAL_RESEARCH_REPORT.md`

所有 JSON 结果放入：

`reports/JEV_PHASE8/`

必须包括：

- STORAGE_BEFORE / AFTER
- CHECKPOINT_INVENTORY
- SAFE_DELETE_MANIFEST
- STORAGE_CLEANUP_REPORT
- CORRECTIVE_OPPORTUNITY_AUDIT
- NATIVE_CORRECTIVE_V2_AUDIT
- LEARNING_CURVES
- DATA_SCALING
- CAPACITY_ABLATION
- SAMPLING_LOSS_ABLATION
- ARCHITECTURE_ABLATION
- NATIVE_PARITY
- HELDOUT_RESULTS
- GO_NO_GO

实际产物只在对应阶段确实完成时标记 COMPLETE。

未运行必须写 NOT_RUN。

失败必须写 FAIL / BLOCKED。

不得填写占位正向结果。

每个阶段保留：

- 精确运行命令
- 数据分割
- Seed
- Source SHA
- Checkpoint SHA
- 日志与失败原因
- 空间使用记录
- WHAT DID WE LEARN

---

# 13. 执行权限与优先顺序

你可以自主完成任务，不需要在每次普通代码修改、单元测试或小规模训练前等待确认。

但不得绕过硬 Gate。

严格按以下顺序执行：

**第一步：存储空间审计，安全清理完全无用的 checkpoint。**

**第二步：冻结 Phase VIII 研究目标与数据划分。**

**第三步：找到真实可纠错 MATCH 事件，量化可行关联上限。**

**第四步：构建小规模真实 native 因果训练数据。**

**第五步：判断数据量、样本分布、训练轮数、容量和监督目标各自的影响。**

**第六步：在 Gate 允许时构建 Candidate-Conditioned JEV。**

**第七步：与强 MLP、DeepSets、固定规则和已有 B2 进行公平实验。**

**第八步：在独立真实在线闭环上判断 GO / NO-GO。**

**第九步：有条件地开展 MEMORY / REACTIVATION 的独立监督研究。**

**第十步：输出完整科研总结及下一阶段授权建议。**

如果中途发现数据、监督、代码契约存在问题，应先解决实际阻塞，不允许以增加模型大小、训练轮数或 GPU 使用量绕过问题。

所有已经完成的历史负结果都属于科研资产，必须保留科学结论。

## 最终必须回答

1. JEV 之前为什么没有超过固定规则？
2. 是不是因为样本太少？
3. 是不是 ACCEPT 占比过高？
4. 是不是训练不够？
5. 是不是网络容量不够？
6. 是不是结构与真实身份选择任务不匹配？
7. 新的 Candidate-Conditioned JEV 是否真的比强 MLP / DeepSets 更好？
8. 未来三模块 Unified JEV 是否还具有可验证的研究价值？

**FINAL PRINCIPLE**

不要追求做出一个表面上完整的三模块 JEV。

我们要做的是一个能够在真实复杂身份关联中纠错、保留正确身份、降低长期错误传播，并能够通过严格公平实验说明自己为何有价值的在线决策方法。

本次 Phase VIII 的成功标准，是形成一个具有可靠因果证据的研究结论，而不是强制得到 HOTA 正提升。
