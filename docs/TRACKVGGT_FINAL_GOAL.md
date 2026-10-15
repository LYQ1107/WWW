# FINAL GOAL — Track-Conditioned VGGT Internal Priors for Efficient Global Multi-Camera Multi-Object Tracking

## 0. 最高优先级指令：严格执行指定研究设计，不允许擅自换方向

你现在负责一个时间极其紧迫的 WWW 2027 多摄像头多目标跟踪研究项目。

本次任务不是让你重新构思三个创新点，而是让你严格实现下面已经设计好的模型，完成消融、训练、真实视频验证，并根据实验结果进行有限且有依据的调整。

研究主线已经确定：

**Track-Conditioned Visual Geometry for Efficient Global Multi-Camera Tracking**

暂定方法名称：

**TrackVGGT-MOT**

基础模型：GMT（CVPR 2026）。

主数据集：VisionTrack。

技术参考：VGGT（CVPR 2025）和 VGGT-Det（CVPR 2026）。

研究目标：在保持 GMT 检测器和关联主干的前提下，利用历史轨迹查询挖掘 VGGT 内部多视角先验，并通过身份关系蒸馏实现低成本在线全局跟踪。

三个预先确定的研究模块：

1. TCVG：Track-Conditioned Cross-View Geometry。
2. CVTIC：Cross-View Temporal Identity Consistency。
3. IRGD：Identity-Relation Geometry Distillation。

VGGT-Det 的两个关键机制必须分别实现和消融：

- AG：Attention-Guided Query Generation。
- QD：Query-Driven Feature Aggregation。

**不能直接复制 AG/QD 后声称是我们的原创。我们的研究差异必须来自历史身份条件化、跨视角时序对应和身份关系蒸馏。**

### 0.1 绝对禁止

禁止以下行为：

- 不准重新训练 GMT Stage1、Stage2。
- 不准一开始就微调整个 VGGT。
- 不准重新构建大型数据集。
- 不准反复使用官方 TEST 调参。
- 不准随意修改 VisionTrack 原始 GT。
- 不准修改 GMT 官方指标公式。
- 不准把 VGGT-Det 的三维检测头直接移植为 MOT 检测器。
- 不准把 AG 改成普通的随机 token 采样。
- 不准把 QD 简化成四层特征直接拼接。
- 不准把历史 GT 身份输入在线模型。
- 不准把未来帧输入因果在线跟踪过程。
- 不准把离线 VGGT 特征缓存当成正式实时部署能力。
- 不准报告没有真实运行得到的指标。
- 不准为了得到提升修改对照模型的检测框或评测配置。
- 不准因为第一次实验失败，就停止整个研究。
- 不准在没有科学证据时宣布论文创新成功。

任何想偏离本设计的行为，先写出原因、预期解决的问题及对照实验。不要直接实施没有依据的结构变更。

---

# 1. 项目、仓库、数据和现有结果

## 1.1 主仓库

https://github.com/LYQ1107/WWW

原有 VGGT 研究分支：

`research/vggt-geo-stream-mcmt-www27-20261014`

本次方案应以该分支最新经过验证的代码为基础，新建隔离分支：

`research/trackvggt-identity-query-www27-20261015`

首先检查远程分支和 HEAD，不要盲目覆盖现有分支。

已审计过的参考提交：

`861c8710fc76935b1af49c9aa1c7df135a8a2ce8`

如果远程已经产生更新，先阅读新的 commit 和报告，确认没有需要继承的有效修复。

禁止把旧 JEV 决策模型混入这次实验。

## 1.2 参考数据

VisionTrack 可能位于：

`/data/DATASETS/TRACKING/JDE/VisionTrack`

原 GMT Stage2 checkpoint 可能位于：

`/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth`

当前研究环境可能位于：

`/data1/liuyeqiang/WWW_vggt_geo_runtime`

请检查真实存在的路径。

不得因为这些绝对路径不匹配，就重新下载数据和重新训练模型。

## 1.3 现有负结果

必须先阅读：

- `docs/GEO_STREAM_FINAL_REPORT.md`
- `docs/GEO_STREAM_REQUIREMENT_AUDIT.md`
- `docs/GEO_STREAM_DENSE_SEED_STABILITY.md`
- `docs/GEO_STREAM_GEOMETRY_FAILURES.md`
- `docs/GEO_STREAM_REPRODUCIBILITY.md`
- `docs/GEO_STREAM_ACTUAL_DECISIONS.md`
- `docs/WWW_SCOPE_NOTE.md`

此前已经确认：

1. 直接使用 VGGT ROI 特征只能获得很小且不稳定的提升。
2. VGGT 几何位姿没有带来独立的最终跟踪输出变化。
3. 稀疏几何更新无法稳定保留身份关联收益。
4. 使用 DINO 或 GMT 原有特征的轻量关联头，也能获得接近 VGGT 的效果。
5. 普通相机/时间固定先验比当前 VGGT 方案更有效。

因此本次不再重复旧实验。

本次新增内容必须针对：

**历史身份查询引导的局部跨视角对应，而不是普通整图 ROI 平均池化。**

## 1.4 锁定基线

当前严格五场景的因果评测参考：

| 方法 | HOTA | AssA | CVIDF1 | IDSW |
|---|---:|---:|---:|---:|
| GMT B0 | 80.748465 | 81.472706 | 95.057962 | 482 |
| Native Prior | 80.840633 | 81.654465 | 95.181615 | 351 |
| 旧 VGGT B5/B7 | 80.821676 | 81.619531 | 95.147448 | 381 |

这些值属于特定种子和严格 DEV5 配对协议，不是官方完整 TEST 结果。

实际运行时必须根据源文件、checkpoint 和配置重新核验。

Native Prior 是这次必须超越的强轻量对照，不能为了表现新方法而将它从比较中删掉。

---

# 2. 先审计 VGGT-Det 的 AG/QD 实现

参考项目：

https://github.com/yangcaoai/VGGT-Det-CVPR2026

官方 VGGT：

https://github.com/facebookresearch/vggt

请逐项阅读下列源码，不要只看 README。

## 2.1 Attention-Guided Query Generation

主要文件：

`projects/VGGTDet/vggtdet/vggtdet.py`

主要函数：

`attention_guided_prob_fps()`

理解它怎样把注意力值与空间采样多样性结合。

原始算法可以概括为：

Priority(p) = normalized_attention(p) + lambda_dist * normalized_distance(p)

第一批采样点由注意力选择。

后续点兼顾注意力与到已有采样点的距离。

但原始 VGGT-Det 工作在预测三维点云上。

**我们的任务不是三维目标检测，因此不能直接复制三维 FPS。**

我们必须把它改造成：

检测框约束 + 历史身份相关性 + 二维空间覆盖的 token 选择。

## 2.2 Query-Driven Feature Aggregation

重点文件：

`projects/VGGTDet/detr3_models/transformer.py`

重点类：

`TransformerDecoder_Multilevel`

它不是单纯连接四层特征。

实际机制包括：

1. 对四个 VGGT 层分别进行通道投影。
2. 使用可学习的 task/See-Query。
3. 通过 task_head 输出四个层权重。
4. 对权重进行 Softmax。
5. 利用当前权重得到加权后的多层特征。
6. 由 decoder 进一步与目标查询进行注意力交互。

我们的关键改造：

原方法的 See-Query 主要服务于场景级三维检测任务。

新方法的 See-Query 必须由 GMT 的历史轨迹身份状态条件化。

不同历史轨迹应该有能力选择不同的 VGGT 特征层。

## 2.3 多层特征获取

重点阅读：

`vggt/models/aggregator.py`

`vggtdet.py` 的：

- `extract_feat()`
- `get_box_features()`

固定优先研究层：

4、11、17、23。

注意 VGGT 官方版本与 VGGT-Det 分支中的 Aggregator 输出格式可能不同。

必须根据实际源码建立：

`layer_id -> tensor`

的明确映射。

不能把长度为 4 的返回列表误当成 24 层原始索引。

不能把 camera tokens、register tokens 当成普通 patch tokens。

必须检查 patch_start_idx。

VGGT-Det 的注意力图接口不是我们现有 FrozenVGGT Adapter 的标准输出。

如果需要使用 VGGT-Det 的修改版 attention extractor：

- 将其作为独立、固定版本的外部依赖。
- 核对与官方 VGGT 权重是否兼容。
- 做 attention map shape、finite、padding 和空间方向测试。
- 不要偷偷用其他显著性算法代替，然后还把结果称为 VGGT-Det AG。

## 2.4 禁止移植的内容

本次不要移植：

- ScanNet/ARKitScenes 的数据加载系统。
- 三维框检测 Head。
- 3D objectness/classification loss。
- 三维体素和渲染网络。
- 原始点云 FPS CUDA 扩展。
- 整套 MMDetection3D 训练环境。
- VGGT-Det 在室内数据集训练的三维检测权重。

仅复用或合理重新实现 AG、QD 所需的思想及最小模块。

记录第三方代码及权重的许可证和来源。VGGT 官方研究材料有单独许可条款，VGGT-Det 仓库中的 MIT 代码许可不能代替 VGGT 的许可。

输出：

`docs/TRACKVGGT_SOURCE_AUDIT.md`

完成这一阶段以后立刻提交并推送一次。

---

# 3. 整个方法的设计总图

输入：

- 当前双摄像头画面：I_A(t)、I_B(t)。
- GMT 当前检测：D_A(t)、D_B(t)。
- GMT 已经建立的历史轨迹：T_1,...,T_M。
- 每条轨迹已有的外观历史与时间信息。

输出：

- 每个当前检测与历史轨迹的身份对应证据。
- 教师模型的候选匹配分布。
- 蒸馏后的轻量学生匹配分数。
- GMT 原有全局轨迹 ID。

模块数据流：

GMT 历史轨迹
    -> Track Query Encoder
    -> Identity-Guided AG
    -> VGGT 局部多层 Tokens
    -> Track-Conditioned See-Query
    -> Query-Driven Aggregation
    -> Cross-View Temporal Consistency
    -> Teacher Association Scores
    -> Identity Relation Distillation
    -> Tiny Student
    -> GMT 原有 GTA/Hungarian
    -> 全局身份输出

训练阶段允许使用冻结 VGGT 教师。

最终实时学生阶段，不需要完整 VGGT。

注意这里的 VGGT backbone 仍然是冻结且无身份条件的。身份条件化发生在我们新增的 AG/QD 解码模块，不允许将其写成“VGGT 自身已经学会跟踪”。

---

# 4. 第一创新点 TCVG：Track-Conditioned Cross-View Geometry

这是第一优先级研究模块。

先验证这个模块确实有用，再做后续设计。

## 4.1 核心研究问题

在跨摄像头身份关联中，普通 VGGT ROI 池化没有利用历史身份查询。

我们希望让历史身份查询决定：

- 哪些 VGGT 局部 tokens 值得使用。
- 哪些目标区域与历史身份最相关。
- 哪些局部区域可以帮助区分外观相近的不同目标。

必须显式区分：

普通注意力显著性与身份相关性。

VGGT 的原始 attention 并不是身份标注，因此不能假设高 attention 一定对应正确目标。

## 4.2 Track Query Encoder

输入一条 GMT 历史轨迹 j。

从原有 GMT 状态中获得：

- 最近一次有效 ReID/关联特征。
- 历史特征的因果 EMA 或有限窗口摘要。
- 最近观测摄像头。
- 当前与最近观测的时间间隔。
- 可观测轨迹长度。

使用现有 `reid_features`。

不要预设输入维度是 1024 或 1152，必须从实际模型读取并断言。

建议首先使用最近 4 次有效观测。

如果轨迹不足 4 次，则按实际数量聚合。

不要用零填充制造虚假的历史身份。

定义：

h_recent = 最近有效的 GMT 表示

h_long = 历史 GMT 表示的归一化均值或 EMA

track_query = LayerNorm(MLP([h_recent, h_long, delta_t, camera_encoding]))

建议：

- 输出维度 d=128。
- 两层 MLP。
- 只用当前或过去信息。
- 训练时允许使用 GT 生成关联标签，但不能用 GT ID 生成查询特征。
- 轨迹必须来自实际运行的 GMT 状态，而不是训练期直接用 GT 构造的理想轨迹。

初始实验先实现均值、EMA 两种轨迹摘要对照。

只有在有证据的情况下才增加 GRU/Transformer 历史编码器。

## 4.3 当前目标候选

对当前摄像头中的检测目标 i，取得：

- GMT 检测框。
- GMT 原始特征。
- 当前物理 camera_id。
- 当前 frame_id。
- VGGT 当前多视角特征。

首先使用 GMT 已有合法候选集合。

不要新建一个会改变检测召回的候选生成系统。

初始只保留 GMT 关联分数最高的 8 个候选历史轨迹。

但必须统计：

- GT 正确候选在 Top-8 中的比例。
- 正确候选被 Top-8 排除的次数。
- 无正确候选的 unmatched 比例。

如果大量正确候选不在 Top-8 内，先修正候选覆盖，再研究特征。

## 4.4 VGGT Backbone

输入同一时刻两路物理摄像头画面。

默认采用官方 padding 方式，把图像映射到 VGGT 的合法输入尺寸。

初始输入张量：

`images: [1, 2, 3, H, W]`

H、W 根据真实预处理确定；常规使用 518×518。

数值应符合 VGGT 输入的 RGB 和归一化约定，不能重复归一化。

冻结模型：

`requires_grad=False`

`eval()`

`torch.inference_mode()`

获取四层：

`F4, F11, F17, F23`

对原始图像坐标、VGGT 输入坐标、patch 网格坐标建立显式变换。

不要复用旧的错误坐标映射。

## 4.5 身份引导的 AG

不要照搬原始 3D FPS。

对候选检测框先建立一个图像空间的合法 token 区域：

1. 检测框内部区域。
2. 检测框附近适当的上下文区域。
3. 超出原始图像有效内容的 padding 区域必须排除。
4. 背景上下文必须有上限，不能让大量背景 token 淹没目标。

对每个候选 patch p，考虑三种证据：

A(p)：VGGT 内部 attention 重要性。

R(j,p)：历史身份查询 j 与 patch p 的学习型相关性。

B(i,p)：patch 与当前候选检测区域的空间关系。

候选分数定义为：

score(j,i,p)
= alpha * normalized_attention(p)
+ beta * cosine(Wq * track_query_j, Wk * token_p)
+ gamma * box_region_prior(i,p)

空间多样性项单独加在贪心 token 选择过程中，不直接当成身份分数。

第一轮不能凭经验固定所有权重就声称有效。

先建立三个版本：

AG-A：只有检测框 + ROI tokens。

AG-B：检测框 + VGGT Attention + 空间多样性。

AG-C：检测框 + VGGT Attention + 历史身份相关性 + 空间多样性。

其中 AG-C 的身份投影必须使用 TRAIN 学习。

不要用未经训练的随机投影直接判断身份。

建议首先选择最多 24 个 patch tokens：

- 优先保留目标内部约 12 个。
- 最多保留 12 个外围上下文 patch。
- 如果内部没有足够的有效 patch，重新分配配额。
- 小目标 token 不足时，标记 small-target 状态并进入专用回退。

这些数量是第一轮默认值，不是永久最优参数。

同时测试 K=16、24、32 的小规模消融。

空间多样性参考 VGGT-Det AG 的思想，但在二维 patch 坐标中实现。

用已有采样 patch 到候选 patch 的最小空间距离作为多样性奖励。

所有采样必须具有确定性的 tie-breaking，避免随机候选顺序造成无关差异。

输出：

每个候选目标的 selected multi-layer tokens。

建议建立统一结构：

`selected_tokens: [L, K, C]`

其中：

- L=4。
- K 为有效选中 token 数。
- C 是原始对应层通道数。
- 同时保留 padding mask、层索引、原图 patch 坐标及有效性。

不能将不等长 token 列表直接用错误的 padding 参与注意力。

## 4.6 航拍小目标问题

这是这次实验必须解决的风险之一。

旧实验中目标尺寸很小，ROI 可能只有一个甚至零个有效 patch 中心。

必须按目标大小分桶：

- 无有效 patch 中心。
- 1–3 个有效 patch。
- 4–15 个有效 patch。
- 至少 16 个有效 patch。

统计每桶的身份识别能力。

同时提供一个小目标裁剪分支：

1. 以检测框为中心扩大约 1.5 倍上下文区域。
2. 将合法裁剪区域重新采样到 224 或 336 分辨率。
3. 使用相同冻结视觉编码器进行特征提取。
4. 只在有限训练样本中测试，不默认给所有目标增加额外 VGGT 推理。

注意裁剪图像是局部放大观察，不是新的真实物理摄像头。

不得将它当作真实相机位姿输入进行三维解释。

必须比较：

全图 tokens、局部放大 tokens，以及两者组合。

如果小目标放大是主要收益来源，必须再与高分辨率 DINO 裁剪作相同计算条件对照。

## 4.7 TCVG 输出

得到：

`E_ij^AG`

表示历史轨迹 j 与当前目标 i 的身份局部对应证据。

它需要有明确的正负样本语义。

先在 TRAIN 困难候选上评价：

- Recall@1。
- Recall@5。
- MRR。
- Pair AUC/AP。
- 正确匹配的排名。
- GMT 错误但 TCVG 修正的数量。
- GMT 正确但 TCVG 破坏的数量。
- 与 DINO 在相同样本上的互补程度。

不要直接进入完整视频训练。

第一批实验必须确认 TCVG 相对于旧 VGGT ROI pooling 是否真正有收益。

---

# 5. 第二创新点 CVTIC：Track-Conditioned QD + Temporal Identity Consistency

这一模块必须包含两个明确组件：

A. 历史轨迹条件下的动态多层聚合。

B. 跨摄像头、跨时间的身份一致性约束。

二者单独消融，不能混为一个模块。

## 5.1 为什么不能直接沿用 VGGT-Det 的 See-Query

VGGT-Det 已经提出 Learnable See-Query。

我们不能把它简单复制过来再改名。

我们的目标是：

**不同历史身份轨迹拥有不同的信息需求，因此可以动态选择不同深度的 VGGT 特征。**

例如：

- 外观差异非常明显时，可能更多依赖具有较强局部外观信息的层。
- 大视角变化时，可能更多依赖跨视角上下文。
- 严重遮挡或历史证据稀少时，模型可以采用不同的层权重。

上述只是可检验假设，不能预先宣称 VGGT 各层必然满足这些语义。

## 5.2 Track-Conditioned See-Query

建立一个新的可学习基础 token：

`q_see_base`

然后根据轨迹查询生成动态 See-Query：

`q_see_j = LayerNorm(q_see_base + MLP(track_query_j))`

每条历史轨迹 j 都有自己的条件化 See-Query。

通过 MLP 输出四个权重：

`w_j = softmax(MLP(q_see_j), dim=-1)`

要求：

`w_j.shape == [4]`

所有权重非负且和为 1。

可选地，将当前候选检测表示加入 See-Query 形成候选条件化权重，但第一版本只做 track-conditioned，避免过多变量。

## 5.3 四层特征投影

VGGT 每层特征投影到同一维度。

建议：

`Projection_l: C_l -> 128`

四层分别拥有独立投影参数。

每层执行：

`Z_l = LayerNorm(Projection_l(F_l))`

对应空间位置必须正确对齐。

不能把不同层 token 顺序不一致的特征直接求和。

四层投影结果：

`Z_4, Z_11, Z_17, Z_23`

按照历史身份条件进行融合：

`Z_agg(j,p) = sum_l w_j[l] * Z_l(p)`

重要：

这里的聚合权重依赖于历史轨迹 j，而不是整个场景只有一个固定向量。

必须记录不同轨迹在不同场景中的层权重分布，判断模型是否真正进行条件化选择。

## 5.4 Query-Driven Cross-Attention

输入：

历史 Track Query：

`q_j: [128]`

动态融合后的候选 tokens：

`Z_agg(j): [K,128]`

使用轻量 Cross-Attention：

`u_ij = CrossAttention(q_j, Z_agg(j))`

初始配置：

- hidden dim=128。
- attention heads=4。
- decoder layers=2。
- dropout=0.1。
- FFN hidden dim=256。

不要复制 VGGT-Det 的完整八层大型 decoder。

使用一个简单的候选打分头：

`s_ij_teacher = MLP([q_j, u_ij, gmt_pair_evidence, temporal_features])`

在初始版本中，候选与历史特征只允许来自合法匹配集合。

最后输出：

`teacher_logits: [N_current, N_historical_proposals]`

维度和索引必须与 GMT get_asso 的原始输出一致。

如果从 track-level 分数映射回 proposal-level 分数，必须显式处理一个轨迹对应多个历史 proposal 的情况，避免轨迹长度被意外重复加权。

## 5.5 一定要做的 QD 消融

QD-0：仅用最后一层 VGGT 特征。

QD-1：四层固定平均。

QD-2：四层拼接 + MLP。

QD-3：普通全局 Learnable See-Query（复现 VGGT-Det 风格）。

QD-4：我们的 Track-Conditioned See-Query。

所有设置尽量采用相近模型容量。

如果 QD-4 不能超过 QD-3，就不能宣称历史身份条件化带来了独立提升。

增加一个必要的机制消融：

随机打乱不同轨迹的历史身份查询，在保持当前图像不变的情况下重新计算层权重和关联。

如果打乱查询后几乎没有变化，说明模型可能没有真正利用历史身份。

这种情况下必须先检查实现，而不是直接进入蒸馏。

---

# 6. CVTIC：跨视角时序身份循环

## 6.1 核心目标

只判断当前两路画面中哪个目标更像是不够的。

我们还希望跨摄像头对应关系与历史时间上的身份延续一致。

构造：

`Camera A(t-1) -> Camera B(t) -> Camera A(t)`

其中 t-1 是过去帧，t 是当前帧。

禁止使用 t+1。

不要求每个目标都在全部三个观测集合中出现。

## 6.2 三个关联矩阵

设：

P_AB：A(t-1) 与 B(t) 的关联概率。

P_BA：B(t) 与 A(t) 的关联概率。

P_AA：A(t-1) 与 A(t) 的直接关联概率。

这些矩阵必须使用同一批合法观测和正确的身份空间索引。

通过矩阵乘法形成：

`P_indirect = P_AB @ P_BA`

然后约束：

`P_indirect ≈ P_AA`

但需要处理 unmatched。

不要因为某个目标在 B 摄像头不可见，就强迫它完成 A-B-A 循环。

只有训练标签确认可见且存在合法对应链时，才允许监督该循环。

其余样本应使用相应 mask，或仅参与普通关联训练。

## 6.3 损失设计

主关联损失：

`L_assoc`

对有正确候选的查询，使用 identity-aware listwise loss。

对没有合法历史轨迹对应的检测，使用 unmatched 类别。

注意一条轨迹可能包含多个历史 proposal，不能强迫唯一一个 proposal 成为正样本。

应当将属于同一真实身份的合法正 proposal 共同计入正概率，或者先聚合为 track-level 概率。

困难负样本损失：

`L_rank = max(0, margin + max(negative_scores) - max(positive_scores))`

只在正负候选同时存在时启用。

建议第一轮 margin=0.2。

时间一致性损失：

`L_cycle`

可以采用合法链条件下的循环负对数概率或掩蔽 KL 一致性。

必须确保：

- 分母、NULL/unmatched 状态正确。
- 非法组合不参与归一化。
- 没有可见闭环时 L_cycle=0。
- 不出现 NaN/Inf。
- 不通过未来标签生成在线特征。

总损失：

`L_total = L_assoc + 0.25 * L_rank + lambda_cycle * L_cycle`

其中：

`lambda_cycle ∈ {0, 0.1, 0.2}`

只允许使用 TRAIN 内部的开发划分选择。

先以 0 和 0.1 对照。

如果加入循环约束没有改善正确身份关联，不要继续任意增大权重。

## 6.4 训练顺序

第一阶段：

训练 TCVG + QD，不加入循环损失。

第二阶段：

从第一阶段已固定的模型继续，启用 L_cycle。

第三阶段：

比较两者的完整目标候选排序。

若 L_cycle 在多个真实场景中降低关联正确率，就关闭这一损失，并将其保留为负结果。

不要为完成三个创新点而强行开启有害模块。

---

# 7. 数据准备：必须改变旧的错误实验样本选择

此前旧实验中存在一个非常严重的问题：

第一批简单候选上，GMT 已经获得 100% Recall@1。

这种样本几乎无法检验新模型是否具备独立身份纠错能力。

本次必须首先构建困难候选训练与诊断集合。

## 7.1 数据来源

只能使用已获准的 VisionTrack 原始数据。

优先复用当前仓库中已经存在的 TRAIN 划分和追踪结果。

不要复制原始图片到一个新的大型数据集目录。

只保存：

- 原始文件路径和图像 ID。
- camera_id。
- frame_id。
- 当前检测框。
- 有限历史轨迹摘要。
- GMT 原始候选及分数。
- TRAIN 标签和有效性。
- 少量提取后的目标级教师特征。

## 7.2 困难样本定义

使用 GMT 固定 checkpoint 进行因果推理。

收集：

1. GMT Top-1 匹配错误，但正确候选在合法集合内的情况。
2. Top-1 与 Top-2 分数差距很小的情况。
3. 两个外观相似但身份不同的行人。
4. 跨摄像头候选冲突。
5. 目标短暂遮挡后的重新匹配。
6. 相机运动明显且目标仍然可识别的情况。
7. 无法确认历史身份的 unmatched 检测。

训练集合可以通过真实 TRAIN GT 标注困难错误类型。

但独立开发评测的候选选择必须事先固定，不允许使用开发 GT 来决定哪些样本被模型推理或接受。

必须统计正确身份候选覆盖率。

不能只研究“正确答案已经在候选里”的简单有利子集，同时忽略候选缺失造成的真实错误。

## 7.3 规模控制

第一轮不需要全量数据。

建议先收集：

- 最多 1000 个有效训练困难事件。
- 最多 300 个开发困难事件。
- 每个事件最多 8 个历史候选。
- 每个候选最多 24 个主要 tokens。
- 每个事件保存正负样本及缺失候选状态。

这些是第一轮计算预算。

不足时按实际合法样本数报告，不允许重复同一个事件凑数量。

按完整场景划分 TRAIN fit 与开发集，禁止随机拆同一段轨迹到两个集合造成泄漏。

## 7.4 GT 约束

必须分清：

- 轨迹查询输入：GMT 实际历史特征。
- 教师训练目标：TRAIN GT 身份。
- 开发集评估标签：仅在推理完成后评价。
- 在线测试模型：完全不可访问 GT。

不能将 GT ID 编码为连续可学习的全局类别，然后在开发/测试直接使用。

候选与轨迹身份应以场景、时间、摄像头和运行状态构造稳定主键。

不同场景中的相同数字 ID 不是同一个人。

---

# 8. 教师模型训练流程

采用冻结编码器、训练小模块的方式。

## 8.1 冻结参数

冻结：

- GMT detector。
- GMT backbone。
- GMT 原始 association transformer。
- VGGT backbone。
- VGGT camera/depth/point heads。

实际上本次不需要计算 VGGT 的完整 camera/depth/point 输出。

训练参数仅包括：

- Track Query Encoder。
- AG 身份相关性投影。
- 四层特征 Projection。
- 条件化 See-Query。
- 两层 Cross-Attention Decoder。
- Candidate Score Head。

第一轮目标是控制可训练参数在约 200 万以内。

如果因为输入通道数量需要超过该值，记录实际参数量和原因，并提供同等容量的 DINO 对照。

## 8.2 梯度正确性

因为 VGGT 冻结，可以先用 `torch.inference_mode()` 提取教师所需 tokens。

但是：

- 投影层必须接收正常可训练张量。
- 冻结特征不能导致 QD 无法反向传播。
- 不允许对整个新网络使用 no_grad。
- AG 的离散 Top-K 选择没有普通梯度，训练可学习相关性时必须设计相应的软权重阶段或明确的训练替代。
- 不能把没有可训练路径的随机 query sampler 当作学习模块。

建议先利用软注意力加权训练身份相关性，再用硬 Top-K 进行推理消融。

## 8.3 第一轮训练

建议：

- Optimizer：AdamW。
- 初始学习率：1e-4。
- Weight decay：0.01。
- 有限训练步数：1000。
- 先运行 100–200 steps 检查训练信号。
- 使用梯度裁剪，建议 max norm=1.0。
- 使用 FP32 计算关联损失。
- 冻结视觉编码器可采用支持的 FP16/BF16 推理。

具体 batch size 根据实际显存选择。

不要为了固定 batch size 导致 OOM。

训练期间每隔固定步数检查：

- loss finite。
- 梯度 finite。
- 正负候选 margin。
- 训练与开发 Recall@1。
- See-Query 四层权重分布。
- 模型是否输出接近恒定分数。
- 正确候选是否进入网络。

不能只看训练 loss 下降。

## 8.4 教师核心对照

必须比较：

T0：GMT 原始候选分数。

T1：GMT + Native Prior。

T2：GMT + 旧 VGGT ROI Pooling。

T3：GMT + VGGT-Det 风格 AG（不使用历史身份）。

T4：GMT + 历史身份条件 AG。

T5：T4 + 四层固定平均。

T6：T4 + 全局 See-Query。

T7：T4 + Track-Conditioned See-Query。

T8：T7 + 时序身份一致性。

另外至少包括：

- 相同训练任务的 DINO 特征教师。
- 相近参数量的普通 MLP。
- 只使用 GMT 历史特征的对照。

不能因为 T8 的网络更大，就将参数量差异解释为几何先验收益。

---

# 9. 关键 Gate：教师是否真的有新身份信息？

完成第一轮教师实验后，必须在固定困难开发样本上回答：

1. T4 是否超过 T2？
2. T7 是否超过 T6？
3. T8 是否超过 T7？
4. T7/T8 是否超过相同容量的 DINO 教师？
5. 它们是否能修正 Native Prior 无法修正的真实身份错误？
6. 真实修正是否大于真实破坏？
7. 优势是否集中在单个场景或单个摄像头对？
8. 历史身份查询打乱后，性能是否下降？
9. 更换 VGGT 层后，是否存在可解释的收益变化？
10. 小目标的改善是否只是图像放大带来的？

只有在匹配候选、评价标签与计算条件一致时，才能比较这些方法。

如果发现 VGGT 教师独有纠错数量为零，而 DINO/GMT 简单对照同样有效，那么必须标记：

`NO_UNIQUE_VGGT_TEACHER_EVIDENCE`

不要把它改名成其他创新后继续声称成功。

但是允许继续完成已经预先指定的 CVTIC 或学生对照，判断负结果来自 AG、QD 还是训练目标。

---

# 10. 第三创新点 IRGD：身份关系蒸馏

第三点的原则非常明确：

**教师使用 VGGT，学生在完整在线跟踪中不调用 VGGT。**

我们不再重复此前失败的“每隔 K 帧运行大模型”方案。

## 10.1 教师输出什么

教师对一个真实 GMT 候选集合输出：

`teacher_logits: [N_query, N_candidates]`

同时保留 unmatched 状态。

经过合法归一化，得到：

`P_teacher(identity | current_observation, history)`

这里的 identity 指动态候选轨迹，不是训练集中固定的人物类别表。

教师提供候选之间的相对关系，而不是三维坐标或整张 VGGT 特征图。

## 10.2 学生输入

轻量学生只能使用原 GMT 在线推理中已经存在的信号：

- 当前 GMT association logits。
- GMT 当前目标特征。
- 历史轨迹特征摘要。
- 当前与历史的时间差。
- 真实物理 camera_id。
- 轨迹长度。
- Top-2 关联 margin。
- 当前和历史观测的相对时序信息。

不要把 VGGT 的任何 token、相机位姿、depth map 或离线教师 logits 作为学生在线输入。

不能把训练期缓存暗中装载到 TEST 模式。

## 10.3 学生结构

第一版建议：

`Student MLP: input -> 128 -> 64 -> bounded residual score`

只对合法候选产生残差。

融合方式：

`new_logits = GMT_logits + student_delta`

其中 student_delta 经过有界约束。

建议初始最大绝对残差为 1.0，并从零残差初始化。

必须保留：

- 全局 Hungarian。
- GMT 原始轨迹关联与记忆规则。
- 无匹配候选的合法状态。
- 不变的检测框与图像时间戳。

如果学生关闭或残差为 0，必须逐框、逐 ID 恢复 GMT OFF 结果。

## 10.4 蒸馏损失

使用三种损失：

A. GT Identity Association Loss：

`L_gt`

B. Teacher/Student Relation KL：

`L_kd = T^2 * KL(P_teacher(T) || P_student(T))`

C. Hard Negative Ranking Loss：

`L_rank`

建议：

`T = 2`

第一轮：

`L_student = 0.5 * L_gt + 0.5 * L_kd + 0.25 * L_rank`

这些是默认实验设置。

必须提供：

- GT-only 学生。
- Teacher-only 蒸馏学生。
- GT + KD 学生。
- DINO 教师蒸馏学生。
- GMT 教师或普通训练学生。

不要直接断定加入 KD 就一定改善跟踪。

教师如果不如 GMT，学生模仿教师也可能变差。

必须以真实困难样本错误分析和配对跟踪指标判断。

## 10.5 蒸馏可靠性

可选地根据教师置信度进行蒸馏加权。

但教师置信度必须来自不需要 GT 的预测量，例如 entropy 或 margin。

置信度校准只允许使用 TRAIN 内部监督。

不能在 DEV/TEST 上看 GT 正确性，再决定哪些教师输出用于学生。

## 10.6 最终在线执行

最终部署模式必须只有：

GMT + Tiny Student。

不能加载完整 VGGT 权重。

必须记录：

- VGGT inference calls = 0。
- 新增学生参数量。
- 学生匹配开销。
- 端到端完整系统 FPS。
- P50/P95 延迟。
- GPU 峰值显存。
- 全局关联 HOTA、AssA、CVIDF1、IDSW。

如果学生比 GMT/Native Prior 慢很多，必须优化学生和关联过程，而不是通过不统计相关模块隐藏开销。

---

# 11. 和 GMT 源码的具体连接方式

你必须使用已经确认过的插入位置。

现有文件：

`gtr/modeling/meta_arch/gtr_rcnn.py`

重点方法：

`GTRRCNN.get_asso()`

该方法调用 GMT 原始 Transformer，生成当前 query 到历史 proposals 的 association logits。

现有 GeometryRuntime 已经在这一位置提供适配接口。

本次不要再次重写 GMT 主干。

建立独立的：

`IdentityRelationAdapter`

它接收：

- 原始 GMT logits。
- 当前检测 Instances。
- 当前物理相机与时间。
- 过去已有合法轨迹。
- 学生模型。

返回：

- 相同 shape 的新 logits。

必须在 Hungarian 之前进行合法修正。

需要注意：

原始 GMT 会将 proposal 级分数聚合为 track 级分数。

因此接口必须支持：

- 当前 N 个检测。
- 历史 M 个 proposals。
- proposals 到 track_id 的映射。
- 同一轨迹多个历史观测。
- NULL/unmatched 概率。
- track length 影响的阈值。

不要假设所有 proposal 一一对应唯一 track。

不能改变原始候选排序后忘记同步恢复索引。

## 11.1 在线训练与推理契约

教师数据收集：

允许只读 instrument。

教师训练：

允许使用历史和当前真实图像，但不能使用未来图像。

在线学生：

只允许当前/过去已到达的数据。

真实跟踪 ID 提交：

仍然由 GMT 原有 tracker 完成。

不允许通过教师 GT 直接写入任何全局身份状态。

## 11.2 关键正确性检查

增加以下单元测试：

- OFF raw prediction parity。
- 零残差逐字节预测等价。
- empty query。
- empty gallery。
- 一个目标。
- 一个轨迹多个历史 proposals。
- 跨摄像头同身份。
- 同摄像头身份排他。
- camera_id 不连续。
- camera 顺序变换及规范还原。
- frame_id 非零起点。
- unmatched 正确处理。
- NaN/Inf teacher evidence。
- 多个 GT 对应观测但只属于同一轨迹。
- 未来帧截断一致性。
- 有效 patch 为零。
- VGGT attention 全零或常数。
- 所有 AG token 都位于 padding 区域的异常输入。

任意关键测试失败，不能开始正式训练评测。

---

# 12. 建议新增的文件

不需要一次性创建全部文件。

根据实际实现逐个建立：

`gtr/geometry/track_query_encoder.py`

历史轨迹查询编码。

`gtr/geometry/identity_guided_sampler.py`

VGGT-Det AG 改造版。

`gtr/geometry/track_conditioned_see_query.py`

历史轨迹条件化层权重。

`gtr/geometry/track_qd_teacher.py`

轻量多层 QD 解码器。

`gtr/geometry/temporal_identity_cycle.py`

跨视角时序一致性。

`gtr/geometry/identity_relation_student.py`

轻量学生。

`gtr/geometry/identity_relation_adapter.py`

GMT 接入适配器。

`geo_tools/collect_track_hard_events.py`

收集 GMT 真实困难候选。

`geo_tools/extract_trackvggt_tokens.py`

教师训练所需 token 和 attention。

`geo_tools/train_trackvggt_teacher.py`

教师训练。

`geo_tools/evaluate_teacher_pairs.py`

身份关系初步评价。

`geo_tools/train_relation_student.py`

学生蒸馏训练。

`geo_tools/run_relation_student_online.py`

在线完整视频测试。

`geo_tools/audit_relation_corrections.py`

配对纠错和破坏分析。

必须优先复用已有：

`gtr/geometry/vggt_adapter.py`

`gtr/geometry/roi_feature_extractor.py`

`gtr/geometry/lightweight_head.py`

`gtr/geometry/online_runner.py`

`gtr/geometry/pair_score_adapter.py`

`geo_tools/evaluate_suite.py`

已有日志和评测器。

不能为新项目复制一套与现有实现互相矛盾的评测系统。

---

# 13. 实验执行顺序与时间预算

由于 WWW 2027 截止时间紧张，按照以下顺序执行。

## Phase 0：工程核验与代码审计

目标预算：2 小时。

完成：

- 检查 Git/GPU/数据/checkpoint。
- 读取 VGGT-Det AG/QD 源码。
- 确认冻结 VGGT attention 接口。
- 记录版本和许可证。
- 保证 GMT OFF 正常。
- 创建新分支。

交付：

`docs/PHASE0_SOURCE_AND_INTERFACE_AUDIT.md`

## Phase 1：困难身份事件数据

目标预算：4 小时。

使用现有 GMT 运行结果和只读 instrumentation。

重点建立包含真实错误关联的有限训练与开发集合。

立即报告：

- GT 候选覆盖率。
- GMT 错误数量。
- 可修正错误数量。
- 训练/开发场景隔离。
- 每种错误类型样本量。

如果错误数量过少，优先扩大训练样本的合法场景覆盖，而不是构造假的错误案例。

## Phase 2：AG 最小实验

目标预算：6–8 小时。

实现 AG-A/B/C。

比较旧 VGGT ROI、DINO 与新的身份引导 AG。

不要训练 QD 和学生。

先取得真实困难候选的排名结果。

如果 AG-C 没有独立收益，检查小目标分辨率、身份投影和采样策略。

只允许执行预先限定的合理修复：

- 正确 ROI 映射。
- 小目标放大。
- 合法 attention 归一化。
- token 选择区域。
- 投影训练是否真正有梯度。

不要做无限随机超参数搜索。

## Phase 3：QD + Track Query

目标预算：8–10 小时。

实现四层投影、全局 See-Query 对照和历史轨迹条件化 See-Query。

先跑一个真实 TRAIN 事件和反向传播测试。

再进行训练。

必须保存：

- loss 曲线。
- AG 消融。
- QD 消融。
- Query shuffle 结果。
- 实际模型参数量。
- 真实修正与破坏数量。

## Phase 4：Temporal Cycle

目标预算：4–6 小时。

先只做可见、合法的时序循环训练。

比较 lambda_cycle=0 和 0.1。

只有看到有效信号，才补充 0.2 实验。

如果循环损失使身份匹配恶化，关闭它，不强行保留。

## Phase 5：Identity-Relation Distillation

目标预算：8–12 小时。

使用固定教师：

- 训练 Tiny Student。
- 对比 GT-only、DINO KD、VGGT KD。
- 检查学生是否保留教师增益。
- 真实因果在线评测。
- 记录完整系统速度。

## Phase 6：完整验证与论文准备

完成：

- 严格 DEV5 完整视频。
- 三个学生训练种子。
- 对应固定 GMT 推理随机种子。
- Native Prior 三种子对照。
- HOTA、AssA、CVIDF1、IDSW。
- 完整系统 FPS、P50、P95。
- Teacher/Student 消融。
- WWW 任务相关性分析。
- 最终科研报告。

各时间是工作预算而非完成保证。可以提前结束明确无效的分支，把时间留给最有证据的模型。

---

# 14. 真正的实验优先级

若计算资源和时间不足，优先完成：

P0：GMT + Native Prior。

P1：普通 VGGT ROI。

P2：VGGT-Det AG。

P3：Identity-conditioned AG。

P4：AG + 普通 QD。

P5：AG + Track-conditioned QD。

P6：P5 + Temporal Cycle。

P7：P5/P6 -> Tiny Student。

P8：相同结构的 DINO 教师及蒸馏学生。

尤其不能遗漏 P4 与 P5。

因为这决定了我们新增的 Track-Conditioned QD 是否只是重新实现 VGGT-Det 的现有 QD。

不能遗漏 P8。

因为这决定了 VGGT 是否真正具有区别于普通 DINO 的独立身份信息。

---

# 15. 全视频结果必须如何评价

重要：

困难候选上的 Recall@1 只能作为中间结果。

不能直接代表完整多目标跟踪能力。

最终仍以真实 VisionTrack 完整视频为准。

分别报告：

- HOTA。
- AssA。
- DetA。
- IDF1。
- CVIDF1。
- CVMA。
- MOTA。
- IDSW。

还要记录实际的匹配修正：

- 原 GMT 错误，新方法正确。
- 原 GMT 正确，新方法错误。
- 新方法引入不确定身份。
- 正确候选不在 GMT 候选集合内。
- 匹配分数变化但 Hungarian 决策未变。
- Hungarian 决策变化但最终轨迹没有改善。
- 内存重关联重新改变身份的情况。

不要把长轨迹产生的大量 output pairs 直接当作大量独立修正次数。

需要同时统计真实 solver decision 和最终轨迹指标。

所有结果必须配对到相同 detector membership、相同原始数据和相同推理随机种子。

---

# 16. 实时性标准

请先检查当前实际硬件。

之前 VGGT 实验报告的机器是 V100，不是最初规划的 A100。

不能假设本次仍然有 4 张 A100 40GB。

运行前执行 GPU 型号、显存、占用、驱动与 CUDA 检查。

不得干扰其他项目任务。

训练教师允许占用单独的 GPU。

学生在线推理必须在与 GMT 基线一致的硬件和输入条件下评测。

速度至少报告：

1. 原始因果 GMT。
2. GMT + Native Prior。
3. GMT + Tiny Student。
4. GMT + 在线 VGGT Teacher（如果实际能运行，作为昂贵上界）。
5. 模型推理外的读取和输出成本。

第一阶段效率目标：

- 最终学生 VGGT 调用数为 0。
- 额外学生匹配计算尽量不超过原始 GMT 总时间的 5%。
- 不明显增加完整系统显存。
- 不牺牲原有因果在线属性。
- 尽量达到或超过 GMT 基线整体吞吐量。

不能把 V100 上几组同步帧/秒的系统叫作已经达到 25 FPS 实时。

如需要声称实时，则必须实测满足所宣称的输入帧率、延迟与部署条件。

---

# 17. GPU 和磁盘控制

训练不要从头跑整个 VisionTrack。

VGGT 教师只在有限的训练事件上执行。

避免缓存：

- 全部原始视频。
- 全部 dense VGGT tokens。
- 全部 24 层 feature maps。
- 全量每帧 attention 矩阵。
- 大量重复模型 checkpoint。

教师可以离线生成有限样本的身份候选关系或选中的压缩 tokens。

缓存必须记录：

- 输入 image_id、frame_id、camera_id。
- 选中历史轨迹快照 ID。
- GMT checkpoint SHA。
- VGGT checkpoint SHA。
- 预处理配置。
- AG/QD 版本。
- teacher model SHA。
- 训练/开发范围。

初始教师工作缓存尽量保持在 2GB 以内。

如果超过，优先减少训练事件数量、压缩目标 tokens 或按需提取，而不是无限增加存储。

不得删除旧 JEV、GMT 或其他项目的有效数据和 checkpoint。

---

# 18. 必须防止的科研失败

## 失败一：教师没有独立收益

如果 Track-conditioned VGGT 不如 DINO：

不能声称 VGGT 对身份关联有独特能力。

先检查预先限定的小目标、token 和 query 问题。

修复后仍无收益，停止扩展大型 VGGT 教师。

保留完整负实验结果。

## 失败二：QD 权重没有真正依赖 Track Query

检查：

- 不同轨迹的四层权重是否变化。
- 打乱轨迹历史后是否变化。
- 是否所有权重收敛成常数。
- QD 梯度是否正常。
- 是否误用了同一个全局 See-Query。

如果 track-conditioned 机制没有实际作用，不能将其命名为创新成功。

## 失败三：Temporal Cycle 没有效果

分析：

- 合法闭环样本数。
- unmatched 数量。
- 相机时间对齐。
- 是否错误地强迫目标跨所有摄像头可见。
- 时序路径是否包含实际可验证的身份对应。

必要时关闭 cycle loss。

## 失败四：学生无法学习教师

区分：

- 教师本身错误。
- 教师获得的额外信息不在学生输入中。
- 蒸馏目标过于尖锐。
- 学生容量不足。
- 学生只学会复制 Native Prior。
- 正负候选分布不均衡。

先检查教师和学生在相同训练候选上的关系，而不是立即扩大模型。

## 失败五：离线关联改善但在线轨迹不改善

必须检查：

- 分数是否进入真实 GMT 关联路径。
- proposal-to-track aggregation 是否淹没新增分数。
- 历史轨迹长度的影响。
- Hungarian 是否实际改变。
- memory bank 是否覆盖前面的决策。
- 正确候选是否在当前合法候选集合中。

不能以离线 pair AUC 代替完整视频成功。

---

# 19. 论文创新性审计

请持续回答下面的问题：

Q1：VGGT-Det 已经提出 AG，我们的身份引导 AG 有什么不同？

Q2：VGGT-Det 已经提出 QD，我们的 Track-Conditioned See-Query 有什么新的条件化机制？

Q3：为什么不用普通 ReID、DINO 或 GMT 自身特征完成同一件事？

Q4：为什么历史轨迹需要决定 VGGT 的不同层权重？

Q5：Query shuffle 是否证明模型确实使用历史身份？

Q6：Temporal Cycle 是否提供普通关联 loss 不具备的额外信息？

Q7：所谓 Geometry 是否只是一般视觉语义特征？有没有相应的几何相关消融？

Q8：为什么不直接使用 GMT + Native Prior？

Q9：蒸馏得到的学生是否真的比普通 GT-supervised MLP 更好？

Q10：教师是否使用未来信息或 GT 构造了不合法在线查询？

Q11：学生在线推理是否真的完全不依赖 VGGT？

Q12：实际计算和存储开销是多少？

Q13：方法是否对不同运动摄像头场景有效？

Q14：VisionTrack 的开发场景是否已经被 GMT 主干训练见过？

Q15：为什么这项工作是 WWW 研究，而不仅是一个视觉跟踪方法？

其中 Q1、Q2、Q3、Q8、Q9 是决定本方法创新性能否成立的核心问题。

必须使用实际实验结果回答，而不是文字包装。

---

# 20. WWW 投稿限制

WWW 2027 Research Track：

摘要截止：2026-10-18 AoE。

全文截止：2026-10-25 AoE。

当前时间非常紧张。

可以同步准备研究问题、方法图、实验表和真实摘要草稿，但没有完成的结果不能写成已经证明的结果。

WWW 需要明确 Web 研究相关性。

本项目可以研究：

面向多源在线视频的跨摄像头身份关联和跨视频片段身份检索。

可以复用已有多流检索模块：

`gtr/geometry/retrieval_index.py`

以及相关 `geo_tools/run_live_retrieval.py` 等工具。

加入真实可执行的检索评价：

- 其他摄像头目标片段检索。
- Recall@1。
- Recall@5。
- MRR。
- Query latency。
- 不同摄像头数量下的索引和更新代价。
- 含完整检测与关联成本的多流系统吞吐量。

必须区分本地多流重放与真实互联网部署。

没有真正 Web 相关的科学实验时，不能仅靠标题让论文符合 WWW。

---

# 21. GitHub 和研究记录

本次研究不允许一次运行很多实验，最后只给一份没有来源的总结。

每完成一个 Phase：

1. 保存当前源码。
2. 保存配置和训练参数。
3. 运行测试。
4. 保存真实输出。
5. 记录 checkpoint SHA。
6. 保存实验数据集和划分 ID。
7. 记录所用 GPU。
8. 记录指标、速度及失败原因。
9. Git commit。
10. Push 到当前研究分支。

至少建立：

`docs/TRACKVGGT_FINAL_GOAL.md`

`docs/TRACKVGGT_SOURCE_AUDIT.md`

`docs/TRACKVGGT_ARCHITECTURE.md`

`docs/TRACKVGGT_EXPERIMENT_PLAN.md`

`docs/TRACKVGGT_HARD_EVENTS.md`

`docs/TRACKVGGT_TEACHER_RESULTS.md`

`docs/TRACKVGGT_STUDENT_RESULTS.md`

`docs/TRACKVGGT_REVIEWER_AUDIT.md`

`docs/TRACKVGGT_FINAL_REPORT.md`

以及自动生成的：

`results/teacher_ablation.csv`

`results/student_ablation.csv`

`results/full_tracking_metrics.csv`

`results/runtime_comparison.csv`

`results/identity_correction_audit.json`

`results/experiment_manifest.json`

不能手工填写虚假的指标。

---

# 22. FINAL GOAL

最终研究目标不是简单完成 AG、QD、CVTIC、IRGD 的代码。

真正的目标是：

**证明历史身份查询能否从 VGGT 内部多视角表征中获得普通 DINO/GMT 特征之外的身份关联收益，并将这种收益蒸馏到不依赖 VGGT 的轻量在线跟踪器。**

区分以下状态。

## SCIENTIFIC_SUCCESS

要求：

- 真实跨摄像头身份对应实验产生正向结果。
- 与 DINO、原 GMT、Native Prior 等强对照匹配。
- 证明增益不是仅由多层拼接或模型参数增大造成。
- 在完整视频中获得可复现的关联指标改善。
- 至少完成多个训练/推理随机性对照。

如果仅在一个种子获得微小改善，只能记为 preliminary signal。

## TEACHER_SUCCESS

要求：

Track-conditioned AG/QD 相比普通 AG/QD、DINO 和简单融合，具有可复现的额外正确身份关联证据。

教师可以较慢，但开销必须真实记录。

## STUDENT_SUCCESS

要求：

在没有任何 VGGT 在线调用的前提下，蒸馏学生相比其匹配基线有真实正向结果，并且新增推理开销足够低。

教师成功不自动代表学生成功。

## PAPER_READY

要求：

- 方法创新与 VGGT-Det 有明确区别。
- 完整实验足够充分。
- 主要指标可信。
- 速度描述符合实际。
- WWW 研究相关性有实质支持。
- 论文可以如实回答核心审稿问题。

不能将工程完成自动记作论文成功。

## NEGATIVE_RESULT

如果完整预设实验和有限修复都没有证明独立收益：

保留真实负结果。

不能伪造结果，也不能无限调参。

但是不能在第一个失败阶段直接停止尚未执行的独立科学假设。

必须完成可执行的预设实验范围，或者记录明确外部阻塞。

---

# 23. 现在立即开始

不要只回复计划。

按照以下顺序真正执行。

**第一个任务：**

检查当前 GMT/VGGT 工作树、GitHub HEAD、GPU、数据和模型，建立新的隔离研究分支。

**第二个任务：**

深入阅读 VGGT-Det 的 AG 与 QD 源码，完成最小可复用实现及真实张量 shape 测试。

**第三个任务：**

从 GMT 实际运行状态中构建困难跨摄像头身份事件，建立合法 TRAIN/DEV 对照。

**第四个任务：**

实现 TCVG 的 AG-A/B/C，不加入 QD、CVTIC 或蒸馏，快速检验历史身份引导是否有效。

**第五个任务：**

实现普通 See-Query 和 Track-Conditioned See-Query，完成独立 QD 对照。

**第六个任务：**

尝试 CVTIC，明确其独立收益或者负结果。

**第七个任务：**

只在教师确实提供有效关系证据的基础上，进行 IRGD 教师—学生蒸馏，同时保留 GT-only/DINO 蒸馏强对照。

**第八个任务：**

在 GMT 原始因果在线路径中部署学生，进行完整真实视频评测和速度评价。

**第九个任务：**

整理全部正负结果、模型结构、审稿人问题、WWW 适配性和实际性能，生成最终科研报告并推送 GitHub。

本项目当前唯一推荐的主线是：

**GMT + Identity-Guided AG + Track-Conditioned QD + Temporal Identity Consistency + Relation Distillation**

不要擅自重新引入已经失败的 VGGT 全局三维位姿约束或旧 JEV 决策模型。

不要消耗时间重新训练 GMT。

不要用大量无意义缓存代替真实实验。

不要为了凑三个创新点而保留无效模块。

**优先通过真实的困难跨视角身份纠错证明新机制，然后再追求完整系统提升和低成本部署。**

现在立即执行，并在每个阶段结束时提交、推送完整代码和可复现科研记录。