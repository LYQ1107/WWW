# FINAL GOAL — Ω-Corr：基于 VGGT-Ω 跨视角局部对应的 GMT 身份关联研究

## 0. 最高优先级任务说明

你现在需要在 GitHub 项目 `LYQ1107/WWW` 中启动一条新的研究分支，验证：

**能否利用冻结 VGGT-Ω 中经过跨视角交互的中间特征，建立目标级局部对应关系，并将这种额外证据用于修正 GMT 的真实身份关联错误？**

本次方法暂定名称：

**Ω-Corr: Object-Centric Cross-View Correspondence for Global Multi-Object Tracking**

研究目标不是重新做三维重建，也不是把旧 VGGT 换成 VGGT-Ω 再跑一遍。

必须清楚区分三件事：

1. VGGT-Ω 可以估计相机几何。
2. VGGT-Ω 中间特征是否包含有用的跨视角目标对应信息。
3. 这些信息是否能够真实改变 GMT 的 Hungarian、阈值和最终身份分配。

前一个问题成立，不自动意味着后一个问题成立。

本次核心假设是：

**VGGT-Ω 的跨视角特征交互可能为同一目标在不同摄像头中的局部对应提供额外证据，而这种证据可能与 GMT 原有外观和历史关联信息互补。**

但这只是待验证假设，不允许提前宣布成功。

本次必须研究四个独立组件：

- Ω-C1：跨视角局部 Token Correspondence。
- Ω-C2：双向局部匹配与空间循环一致性。
- Ω-C3：可靠性控制的候选身份分数融合。
- Ω-C4：GMT 不确定性触发的按需 Ω 推理。

其中 C1/C2 是最重要的科学验证，C3 是完整跟踪验证，C4 是效率设计。

先证明 C1/C2 真正有用，再进入 C3/C4。

不能只做一个 F 矩阵替换。

---

# 1. 当前仓库、代码及已完成实验

GitHub：

https://github.com/LYQ1107/WWW

上一轮研究分支：

`research/vggt-omega-geometry-proof-20261016`

已核验的最新参考提交：

`594d9d3237966f1a4d73a3f689eec48e585f0e58`

先检查远程最新 HEAD。

本次新建：

`research/omega-crossview-correspondence-20261016`

使用隔离 Git worktree。

保留以前所有源码、checkpoint、结果和负实验。

不能覆盖旧报告。

必须阅读：

- `docs/OMEGA512_FINAL_RESEARCH_REPORT.md`
- `docs/OMEGA512_EXPANSION_REPORT.md`
- `docs/OMEGA512_PILOT_REPORT.md`
- `docs/OMEGA_FINAL_RESEARCH_REPORT.md`
- `docs/TRACKVGGT_FINAL_REPORT.md`
- `docs/TRACKVGGT_CORRECTABLE_ERROR_ANALYSIS.md`
- `docs/TRACKVGGT_DEV12_NATIVE_ASSET_HANDOFF.md`

## 1.1 以前已经证实的事实

旧 VGGT-Ω 几何实验：

- 512 原版权重已经取得并通过哈希校验。
- 416 官方复现权重仍未获得合法访问。
- 已验证 96 组真实同步相机图像。
- 固定背景对应方案中，Ω 与旧 VGGT 各有 13/96 组共同通过有效性门槛。
- 这 13 组是相同物理图像对。
- 旧 Ω 几何没有证明独立于原 VGGT 的身份收益。
- 实际 GMT 求解反事实实验中，最终身份 ID 变化为 0。
- 没有合法的新完整视频 TrackVGGT 正向指标。

这些结果必须保留为旧几何负结果。

**本次禁止直接重复原来的 F-only 和相机位姿置信度实验。**

本次不再把背景 F 通过作为使用局部 Tokens 的强制前提，因为我们要验证的是另一条独立假设。

但如果声称局部匹配具有“三维几何对应”含义，必须额外提供几何验证，不能把普通视觉特征相似度直接包装成空间几何。

---

# 2. 现有权重与数据必须直接复用

已有官方 Ω512 权重路径：

`/data1/liuyeqiang/WWW_omega_runtime_20261016/weights/vggt_omega_1b_512.pt`

已记录 SHA256：

`c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`

固定官方源码提交：

`48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6`

已有源码位置：

`/data1/liuyeqiang/WWW_omega_runtime_20261016/official_vggt_omega`

实际运行前重新核验路径和 SHA。

如果已存在正确文件，不要重新下载。

416 官方复现模型继续记为：

`BLOCKED_WEIGHT_ACCESS`

不因此停止独立的 512 实验。

绝对不允许使用随机权重、未经核验的模型文件或者 GGUF 量化转换结果冒充官方 Ω512。

## 2.1 现有数据

VisionTrack：

`/data/DATASETS/TRACKING/JDE/VisionTrack`

GMT Stage2：

`/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth`

已有完整 TRAIN 因果跟踪审计：

`/data1/liuyeqiang/WWW_trackvggt_runtime/full_train_audit_v1`

已封存 DEV12 原生资产：

`/data1/liuyeqiang/WWW_trackvggt_runtime/cv_pil_dev12_native_release_v1`

当前 Ω 研究运行目录：

`/data1/liuyeqiang/WWW_omega_runtime_20261016`

不要重新跑完整 12 条 TRAIN 的 GMT 关联审计。

不要重复保存原始图片。

不要重新训练 GMT。

---

# 3. 先理解 VGGT-Ω 的真实特征结构

必须阅读官方：

`vggt_omega/models/aggregator.py`

`vggt_omega/models/vggt_omega.py`

`vggt_omega/models/heads/camera_head.py`

`vggt_omega/models/heads/dense_head.py`

以及当前：

`gtr/geometry/omega_adapter.py`

`gtr/geometry/omega_geometry.py`

特别注意：

VGGT-Ω 不是简单输出普通 DINO features。

官方 Aggregator 包含：

- Frame Attention。
- Inter-frame Attention。
- Global / Register Token interaction。
- 四个被缓存的中间层。

默认缓存层：

`[4,11,17,23]`

官方返回：

`aggregated_tokens_list`

以及：

`patch_token_start`

Ω 的默认 patch_size=16。

输出中相应层通常具有：

`[B, N_views, N_tokens, 2048]`

其中最后 2048 维是两个 1024 维部分的拼接：

- 当前 block 的 frame-attention 输出。
- 当前 block 的跨帧注意力输出。

**重要：frame-attention 半部并不保证完全没有跨视角信息，因为前面 block 已经进行过跨帧交互。**

因此必须添加真正的单视图独立前向作为控制，不能仅将特征前 1024 维称为纯单摄像头表示。

另外：

`patch_token_start=17`

通常对应 1 个 camera token 与 16 个 register tokens。

不能把这些特殊 token 混入目标 ROI 对应图。

## 3.1 提取方式

建议新增：

`gtr/geometry/omega_correspondence_encoder.py`

输入：

`images: [1,2,3,H,W]`

输出：

- Layer4 patch features。
- Layer11 patch features。
- Layer17 patch features。
- Layer23 patch features。
- 有效图像区域 mask。
- 原图到 Ω 网络坐标映射。
- 实际 patch 网格坐标。
- 物理 camera ID。
- 真实 timestamp。

优先只保留 Layer11、17、23 进行第一轮实验。

Layer4 作为补充消融。

不允许对全部 24 层持续缓存。

优先复用当前 OmegaAdapter 的官方预处理、权重加载和设备逻辑。

如果增加 hook：

- 不得改变原始模型输出。
- 不得修改官方模型权重。
- 必须测试相同输入、相同数值后端下的官方输出一致性。
- 如现有 FP16 后端本身不具有 bitwise 重复性，记录误差并用受控数值容差测试，不得把本来存在的浮点波动当作机制有效性。
- 真实训练与推理必须记录数值精度。

第一阶段冻结 Ω 全部参数。

---

# 4. 必须修复的核心问题：航拍目标很小

此前 TrackVGGT 已经显示：

原始验证 300 个查询中：

- 171 个没有有效 patch 中心。
- 127 个仅覆盖 1–3 个 patch。
- 只有 2 个覆盖 4–15 个 patch。

因此不能直接规定“一个人取 8 个点”，然后实际用同一个 patch 特征复制八遍。

这是错误的点级对应实验。

本次必须保留**真实空间分辨率**。

## 4.1 原始目标映射

对原始 GMT bbox：

`bbox_xyxy_original`

使用当前已验证的 OmegaAdapter PixelMapping。

映射到网络输入：

`bbox_xyxy_omega`

建立：

- 真正落在目标框内的 patch centers。
- 上下文范围内的 patch centers。
- 有效图像 mask。
- padding mask。
- crop mask。

每个 token 必须记录真实空间位置。

不得把无效 padding 当成对象证据。

## 4.2 目标 Token 选择

第一轮实现：

`K_max = 16`

但实际有效 tokens 数量由目标真实尺度决定。

如果只有一个不同的 patch，那么：

`N_valid = 1`

不能复制出 16 个虚假的独立点。

必须记录：

- 原始 bbox 像素面积。
- 缩放后的 bbox 尺寸。
- 目标框内有效 patch 数量。
- 扩展区域有效 patch 数量。
- patch 的图像坐标。
- patch 是否实际属于目标框。
- 是否使用上下文。

建议分桶：

`0`

`1`

`2–3`

`4–8`

`9+`

对于小于两个有效 token 的情况：

不允许报告多点循环匹配成功。

要么标记 `POINT_CORRESPONDENCE_UNAVAILABLE`，要么仅参加已声明的单特征/ROI 对照。

## 4.3 上下文模式

固定三个版本：

M0：只使用检测框内部真实 patch。

M1：使用检测框 1.5 倍范围内的 patch，但区分目标 token 与背景 token。

M2：只使用周围背景环形区域。

M2 必须作为负控制。

如果 M2 与目标区域一样有效，说明模型可能主要利用了背景/场景关联，而不是真正的目标局部对应。

不能把背景信息产生的候选偏差称为人物几何身份匹配。

---

# 5. 创新假设一：Cross-View Token Correspondence

这是本次第一核心实验。

输入两幅同步摄像头图像：

`I_A(t)`

`I_B(t)`

以及 GMT 当前合法的目标候选：

- A 中已经存在的实际参考 proposal。
- B 中当前待关联 detection。

只比较**同一时刻、真实已发布、原生 GMT 合法候选**。

不要构造 GT 完美历史。

## 5.1 Token 序列

对当前目标 i：

`X_i = [x1,...,xM]`

对参考目标 j：

`Y_j = [y1,...,yN]`

每个 token 均来自 Ω 的真实空间 patch。

维度应从官方模型实际输出读取。

对于第一轮使用的 2048 维拼接特征：

`x = normalize(x, dim=-1)`

`y = normalize(y, dim=-1)`

建立：

`C_ij = X_i @ Y_j.T`

得到：

`C_ij.shape = [M,N]`

矩阵元素为跨视角局部视觉特征的相似性。

这一矩阵还不是几何真值。

不得假设不同视角的人体每一个 patch 都观察到了相同三维表面。

## 5.2 必须区分三种特征

实现：

F0：Frame-half。

F1：Inter-frame-half。

F2：完整拼接特征。

另外真正独立执行：

F3：两张图分别单独输入 Ω 后提取的特征。

对 F0/F1/F2 在相同联合输入下比较。

对 F3 使用分别推理的单视图特征。

必须记录 F3 的额外模型调用及计算成本。

研究关键：

**联合多视角输入是否确实让目标局部对应比独立单视图特征更好？**

如果 F3 不比 F1 差，不能直接声称跨视角几何交互提供了独有信息。

## 5.3 Mutual Nearest-Neighbor

对于 C 矩阵，计算两个方向的局部最佳匹配。

A 到 B：

`nearest_B[m] = argmax_n C[m,n]`

B 到 A：

`nearest_A[n] = argmax_m C[m,n]`

满足：

`nearest_A[nearest_B[m]] == m`

的才是 mutual match。

计算：

- Mutual matching 数量。
- 双向匹配覆盖。
- Mutual cosine mean。
- 全部 patch 覆盖比例。
- 不同 token 之间的分数 margin。
- 无法产生有效对应的比例。

不允许仅挑选最高的一两个匹配点，再忽略大量不匹配 patch。

需要将分子和总有效 patch 数量一起报告。

---

# 6. 创新假设二：Cross-View Cycle Consistency

当前设计的循环是同一时刻的空间循环：

`A(t) -> B(t) -> A(t)`

它与之前跨时间的 CVTIC 不是同一种实验。

本次不要加载旧的时序 Cycle 模型。

使用 Ω 当前图像 pair 的局部特征相似矩阵完成双向匹配。

计算：

- 单向最近邻。
- 双向 mutual 最近邻。
- 软双向对应。
- 一对一局部匹配对照。

一对一匹配可以使用 Hungarian，但它仅用于**局部 token 对应**，不能替代 GMT 的轨迹级 Hungarian。

## 6.1 循环有效条件

至少有足够数量的不同真实空间 patch。

不能出现：

- 只有一个 patch，仍宣称 cycle=100%。
- 大量重复位置被当作独立点。
- padding patch 参与循环。
- 匹配点只落在相邻背景而没有目标 token。
- 自己跟自己比较后宣称跨视角对应。

必须明确标注循环对应是：

`OBJECT_REGION`

还是：

`BACKGROUND_CONTEXT`

不允许合并。

## 6.2 组合分数

先不要训练复杂网络。

构建一个确定性的候选对应证据：

`S_omega = f(mutual_similarity, mutual_coverage, cycle_consistency)`

建议第一轮输出一个向量，而不是立即融合成单个分数：

- mean_mutual_similarity。
- mutual_coverage。
- cycle_consistency。
- best_vs_second_margin。
- valid_token_count。
- object_token_fraction。

将这些指标分别保存。

必须先证明哪些指标有实际判别力，再考虑组合。

不要随意拟合六个权重并用相同数据评价。

---

# 7. 一定要有独立几何验证

非常重要：

**Mutual matching 或 A→B→A 自洽，不能自动证明几何对应正确。**

不同的视觉特征也能产生完全自洽的错误循环。

因此新增：

`Geometry Support`

研究对象不仅要有目标对应，还要有独立的空间支持。

## 7.1 可选的几何验证

若当前图像对已有可靠 F：

- 计算互相匹配 patch 的极线残差。
- 记录跨视角局部位置是否合理。
- 检查相机方向转换。
- 检查是否位于有效图像范围。

若 F 不可靠：

- 仍保留视觉对应实验。
- 但该候选的 `geometric_support` 记为未知或无可靠证据。
- 不能通过错误 F 否定所有 Ω 局部匹配。
- 不能因为局部匹配高，就反过来宣称 F 已经可靠。

区分：

`VISUAL_CORRESPONDENCE_EVIDENCE`

与：

`GEOMETRICALLY_VERIFIED_CORRESPONDENCE`

这两个名称不可混用。

## 7.2 原始目标框与三维表面问题

人体两侧视角可能看到不同的身体表面。

所以：

目标级身份相同，不代表每个局部 patch 有一对一的物理点对应。

不能只用是否满足精确像素一致来判定身份模型失败。

需要同时评价：

- 目标级正确候选排序。
- 局部匹配覆盖率。
- 已知背景几何支持。
- 遮挡/不同视角下的失效情况。

---

# 8. 必须完整实现的强对照

这次最重要的科研问题是：

**为什么一定需要 VGGT-Ω，而不是旧 VGGT、DINO 或 GMT 自己的外观特征？**

所以必须在相同的物理目标候选集合上比较：

B0：GMT native appearance / GTA。

B1：GMT + Native Prior。

B2：旧 VGGT ROI cosine。

B3：旧 VGGT patch-level mutual matching。

B4：DINO patch-level mutual matching。

B5：Ω 单视图独立 features + point matching。

B6：Ω 联合输入的 frame-half point matching。

B7：Ω 联合输入的 inter-frame-half point matching。

B8：Ω 完整跨视角 tokens + mutual matching。

B9：B8 + 双向循环一致性。

B10：B9 + 独立可靠几何支持。

另外添加两项负控制：

N1：同一场景中，将错误的物理摄像头图像作为 Ω 配对输入。

N2：保持真实图像、真实目标集合不变，但打乱候选目标与 Token 区域的对应关系。

负控制必须保持合法物理时间或明确声明是人为安慰剂。

不要在负控制中读取 GT 身份作为模型输入。

如果 B8/B9 的效果与 B4/DINO 类似，不能声称 VGGT-Ω 具备独有跨视角身份信息。

如果 B8 与 B5 几乎一样，不能声称联合输入提供了独立的跨视角交互收益。

如果 N1/N2 与原始分数几乎一样，应该怀疑特征没有真正利用配对目标的空间位置或视角信息。

---

# 9. 数据实验必须使用真实 GMT 错误

不得重新用简单均匀采样的几十条身份查询完成全部科学评价。

当前已有完整 TRAIN 审计。

原先 12 条关联训练序列中存在：

- 169468 次 GMT 主关联查询。
- 1553 次已认证主关联错误/漏关联。
- 650 次错误具有同步对侧原生参考。
- 314 次具有同步纯净正确候选。
- 其中 297 次的同步正确候选在 Top8 内。

这些数据已经有现成审计。

不需要重新运行完整 GMT。

## 9.1 主要样本

重点提取：

A. 已知 GMT 真实错误，且存在合法同步纯净正例的事件。

B. GMT 正确，但外观 margin 很低的事件。

C. 相机视角变化较大但当前目标有合法对应的事件。

D. 正常且容易的正确关联对照。

E. 无同步候选或没有可认证纯净正例的失败/未知事件。

对于 E 类，必须计入真实适用性分母，不能将其删除后声称整体正确率很高。

## 9.2 训练与验证隔离

数据仅来自原始 TRAIN。

第一轮建议按完整场景区分：

Fit 场景：

`[4,5,6,7,8,20,24]`

独立机制检查场景：

`[9,13,17,21]`

Scene12：

额外已接触标签的诊断，不用于初次阈值选择。

必须在运行新的 Ω correspondence 结果之前，将这些角色固定到 manifest。

如果某个固定验证集合可认证身份错误不足，只报告样本不足。

不得查看 Ω 结果后，任意把有提升的场景改成验证场景。

所有这些场景仍属于原始 GMT 主干已见过的训练数据，不得称为全新 backbone-unseen 泛化测试。

## 9.3 首轮模型计算预算

优先复用之前的 96 个已提取 Ω512 图像对。

先不要重跑这些模型前向。

核验旧文件是否有中间 Tokens：

- 如果已有完整合法、哈希匹配的目标 Tokens，直接复用。
- 如果只保存了相机参数，则在同一固定图像对上重新执行一次 Token 提取。
- 不得重复执行相同的无效 Pose/F 扫描。

第一轮最多选取约 96–160 组跨摄像头图像对，并记录所有真实关联候选。

但有一个关键约束：

**不要因为此前 13/96 个图像对通过 F 门槛，就只研究这 13 组。**

本次局部视觉对应是假设不同的证据来源，需要在可靠 F、无可靠 F 两种情况下分别评价。

若原有 96 组无法覆盖足够多的实际可修正身份事件，可以根据预先固定的 TRAIN 错误索引，追加有限图像对。

新增对象必须在使用 Ω 新分数前确定。

---

# 10. 先做冻结模型证明，不要立即训练

Phase A：

只执行冻结 Ω、旧 VGGT、DINO 和原生 GMT 的比较。

不训练新的 AG、QD、Transformer。

不重新微调 GMT。

不微调 VGGT-Ω。

必须获得：

- 有效 token 覆盖。
- 有效候选数量。
- 匹配正确率。
- 正负匹配 margin。
- Mutual match 数量。
- Cycle consistency。
- 相比 GMT 的已知排序修正。
- 相比 GMT 的已知排序破坏。
- 相比 DINO 的独立修正。
- 相比旧 VGGT 的独立修正。
- 按场景、人物、目标尺寸分组的结果。

如果 Ω 完全没有独有候选区分信息，不允许宣称已经验证了 Ω correspondence expert。

不过，不能在仅一个默认层结果为负时直接停止预先规定的其他独立比较。

应完成固定的 Layer11/17/23、联合/独立输入、有效 ROI 与上下文控制，然后再得出本轮结论。

---

# 11. 小目标补救实验

这个部分是必要实验，而不是随便增加的新创新点。

如果统计发现：

`N_valid_patch < 2`

比例很高，则执行有限的局部放大对照。

固定：

- 全图原始 Ω512。
- 检测框 1.5 倍上下文局部 crop。
- 局部 crop 放大至 224。
- 局部 crop 放大至 336。
- 相同裁剪与预算条件的 DINO。

注意：

局部放大产生更多编码 patch，不代表增加了原始图像中的真实空间细节。

局部 crop 不能作为新物理摄像头参与原始相机几何计算。

其作用只是验证原始输入缩放是否丢失了目标级视觉信息。

必须分开报告：

`FULL_IMAGE_MULTIVIEW`

与：

`LOCAL_CROP_VISUAL`

两条路线的结果。

如果放大之后 DINO 同样有效甚至更好，就不能将收益归因于 Ω 的几何表征。

---

# 12. 何时允许训练轻量关联模块？

只有冻结模型结果证明存在可利用的候选区分证据时，才允许进入训练阶段。

这里不是训练新的大模型。

只训练一个很小的：

`Correspondence Calibration Head`

输入：

- GMT 原始关联分数。
- Ω mutual matching 相似度。
- Ω matching coverage。
- Ω cycle consistency。
- 有效 token 数。
- 候选空间几何可靠性。
- 原生 GMT 的合法时间与摄像头信息。

输出：

`delta_association_logit`

建议：

`input_dim -> 32 -> 16 -> 1`

并通过有界函数限制关联残差。

最大残差绝对值初始设为 1.0。

零残差初始化。

不允许使用历史 GT 身份向量或未来图像作为推理输入。

## 12.1 训练目标

训练时使用原始 TRAIN 的可靠身份标签。

每行 association query 必须区分：

- 正确候选。
- 错误候选。
- mixed/unknown。
- unmatched。

Mixed/unknown 不能强行当作负样本。

如果一个轨迹有多个历史 proposals，要尊重 GMT 原始轨迹聚合规则。

先使用 identity-aware listwise loss。

再尝试有限 hard-negative margin。

建议首轮：

- AdamW。
- 学习率 1e-3。
- Weight decay 1e-4。
- 训练步数 500。
- 最多固定 3 个训练种子。
- 不更新 Ω/GMT 参数。

不得在 DEV5 或官方 TEST 调参。

## 12.2 学习对照

必须训练同任务、相近参数量的：

- Native-only MLP。
- DINO correspondence MLP。
- 旧 VGGT correspondence MLP。
- Ω correspondence MLP。

每组保留相同合法候选及监督集合。

如果 Ω 由于小目标 token 无效导致参与训练的样本变少，必须同时提供共同有效样本比较与完整样本覆盖比较。

不能把不同有效样本分母的结果混在一张表中宣称模型优劣。

---

# 13. 关键：必须进入真实 GMT solver 才能算最终有效

不要把局部匹配分数提升直接称为多目标跟踪提升。

真正插入的位置：

`GTRRCNN.get_asso()`

原流程：

`GMT Transformer -> association logits -> GTA / Hungarian`

新流程：

`GMT Transformer -> 原始 logits -> Ω 有界 residual -> GTA / Hungarian`

禁止修改：

- GMT Backbone。
- Detection Head。
- ReID。
- RPCE。
- GTA 参数。
- Hungarian 实现。
- 轨迹 ID 管理。
- 原始 memory bank。

## 13.1 合法修正条件

当前第一版只允许作用于：

- 同一时刻的 A/B 两路图像。
- 当前已经到达的图像。
- 已经在原始 GMT 中合法存在的参考 proposal。
- 原始 GMT 当前合法的关联候选。
- 当前已通过对应证据有效性检查的候选。

不允许对未来帧进行查询。

不允许将不同独立 Ω forward 的三维世界坐标直接合并。

不允许把旧图像中的人物框用当前 F 当作当前空间点。

## 13.2 必须保留实际状态分叉

优先复用：

`geo_tools/omega_512_solver_forks.py`

它已经实现：

- 原 GMT 连续因果前缀。
- 当前组状态分叉。
- RNG 恢复。
- 原始检测复用。
- 原始 GTA/Hungarian。
- Bank/ID 状态验证。
- 原生 OFF 对照。

本次不重新写一个假的 Hungarian 代替 GMT。

把 Ω-Corr 分数接入当前状态分叉。

在相同原始输入与历史状态下，对比：

G0：GMT OFF。

G1：GMT + Native Prior。

G2：GMT + DINO correspondence。

G3：GMT + 旧 VGGT correspondence。

G4：GMT + Ω correspondence。

G5：G4 + cycle evidence。

G6：G5 + calibrated reliability。

记录每个状态：

- 原始 proposal Top-1。
- 真实 GMT track-level 排序。
- Hungarian 实际匹配。
- Strict threshold 是否接受。
- 是否生成新 ID。
- Memory Bank 是否改写。
- 最终提交 track ID。
- GT 事后认证的纠正。
- GT 事后认证的破坏。
- 无法认证的变化。

特别注意：

**候选 Top-1 改变，不等于 Hungarian 决策改变。**

**Hungarian 决策改变，也不等于最终身份正确。**

必须完整记录三个层次。

---

# 14. 对应证据如何触发 Ω？

这属于后续效率实验。

不能在第一阶段就实现一个复杂的智能调度器。

先固定一个简单的 GMT uncertainty trigger。

可用观察量：

- 原始 track-level Top1/Top2 margin。
- 原生关联分数 entropy。
- 已合法到达的同步对侧候选数量。
- 原生 GMT 当前匹配冲突。
- 当前轨迹是否具有可用历史外观。

触发条件不能依赖 GT。

第一版固定少量触发预算：

- 所有合法同步候选：用于精度上界和成本测量。
- 原生不确定性最高的约 10%。
- 原生不确定性最高的约 20%。
- 固定时间间隔调度作为计算预算对照。

百分比按实际可调用的同步组或查询定义，不能混用。

因为 Ω 推理需要整个图像对，多个困难目标在同一个时刻应共享一次 Ω 前向。

不能每个困难候选都重新运行一次 VGGT-Ω。

每个同步组最多一次正向 Ω 调用作为默认部署设计；需要额外 AB/BA 调用时必须报告两次实际成本。

## 14.1 不能直接把 Ω 当实时模块

当前 Ω512 模型较大。

必须记录：

- 每个图像对实际前向延迟。
- Ω 调用总次数。
- 每次调用服务的候选数量。
- GMT 原生速度。
- 总系统速度。
- P50/P95。
- 额外 GPU 显存。
- 额外模型驻留内存。
- 跳过 Ω 的比例。
- 几何/对应结果有效利用比例。

不能用离线缓存速度冒充真实在线运行。

如果增加 Ω 后速度非常慢，即使身份有提升，也必须诚实报告。

按需调用本身不是原创创新，必须证明其精度—成本权衡。

---

# 15. 必须定义清楚的研究 Gate

## Gate A：真实 Token Evidence

要求：

- 官方真实 Ω 权重。
- 合法实际 patch token。
- 正确坐标与 mask。
- 足够数量的有效对象 token。
- 相同候选上的正确/错误对照。
- 比普通随机匹配和简单视觉特征具有可检验的区分力。

Gate A 不通过，不允许声称已发现 Ω 的几何身份信息。

## Gate B：Unique Cross-View Evidence

要求：

- Ω 联合输入优于或提供额外的、可靠的单视图对应信息。
- 与 DINO、旧 VGGT 的独立对照有效。
- 匹配收益不只是背景共现。
- 已知净排序纠正超过已知破坏。
- 在预先固定的场景中重复出现，而不是仅集中于一个人物。

如果没有独有收益，应记录：

`NO_UNIQUE_OMEGA_CORRESPONDENCE_EVIDENCE`

## Gate C：Actual GMT Identity Gain

要求：

- 至少存在实际 GMT solver 正确改变。
- 已知真实纠正大于已知破坏。
- 不是单纯修改未进入实际分配的 logit。
- 不依赖 GT 或未来帧。
- 不靠修改 Native Prior/基线制造优势。

若未改变实际求解，则标记：

`NO_NATIVE_SOLVER_IDENTITY_GAIN`

## Gate D：Full Tracking Gain

只有 Gate C 提供可信净收益，才进入完整视频评测。

比较：

- GMT。
- GMT + Native Prior。
- GMT + 最强有效 Ω-Corr。
- 相同结构的 DINO/旧 VGGT correspondence control。

至少报告：

- HOTA。
- AssA。
- CVIDF1。
- IDSW。
- DetA。
- IDF1。
- MOTA。
- FPS。
- P95。
- 模型调用次数和显存。

不得将全视频未运行写为 0 增益。

应写：

`NOT_EXECUTED` / `NOT_MEASURED`

---

# 16. 预设修复与停止条件

这次不要第一次 NO_GO 就立即结束所有独立机制，但也不能无限随机实验。

允许的有限修复：

1. 实际 patch 坐标映射错误。
2. 特殊 token 没有正确去除。
3. 原图 crop/pad mask 错误。
4. FP16 数值导致非有限特征。
5. 对象 token 被背景覆盖。
6. 只有零/一个有效 patch。
7. 联合与单视图输入接口错误。
8. 原始 GT 纯净候选和 mixed 历史混淆。
9. Proposal-to-track score mapping 错误。
10. 当前 Ω evidence 没有真正进入 GTA 求解。

如果遇到这些问题，必须优先修复工程错误，然后重复受影响的最小实验。

但不能通过：

- 修改 GT。
- 选择最容易的场景。
- 删除失败样本。
- 看 DEV5 调参数。
- 使用未来帧。
- 更换未经核验的权重。
- 不断增加网络层。
- 反复降低正向 Gate。

来制造科研成功。

如果 C1、C2 及全部预设对照都没有给出有效身份信息，停止大规模训练和完整视频重放，但保留全部负结果。

---

# 17. 建议代码文件

优先扩展：

`gtr/geometry/omega_adapter.py`

`gtr/geometry/omega_geometry.py`

`geo_tools/omega_common.py`

`geo_tools/omega_512_solver_forks.py`

新增：

`gtr/geometry/omega_correspondence_encoder.py`

负责冻结 Ω 的多视角 patch token 提取。

`gtr/geometry/object_patch_regions.py`

负责原图检测框到 Ω patch 坐标的合法映射。

`gtr/geometry/crossview_point_matcher.py`

负责 mutual / spatial cycle / coverage。

`gtr/geometry/omega_relation_head.py`

负责可选的轻量候选分数校准。

`geo_tools/omega_corr_prepare_cohort.py`

固定真实候选研究集合。

`geo_tools/omega_corr_extract.py`

对固定图像对提取选定 tokens。

`geo_tools/omega_corr_evaluate.py`

计算正负候选对应结果。

`geo_tools/omega_corr_controls.py`

统一旧 VGGT、DINO、单视图及负控制。

`geo_tools/omega_corr_solver_forks.py`

复用实际 GMT 决策状态分叉。

`geo_tools/omega_corr_train_adapter.py`

仅在预设证据成立时训练小型校准头。

`geo_tools/omega_corr_online.py`

仅在通过身份 Gate 时进行完整视频在线测试。

`geo_tools/omega_corr_finalize.py`

生成最终审计和结论。

实际命名可以适应现有仓库，但不得生成第二套互相矛盾的 GMT 评测器。

---

# 18. 产物与报告

每个阶段必须提交并推送 GitHub。

必须生成：

`docs/OMEGA_CORR_FINAL_GOAL.md`

`docs/OMEGA_CORR_SOURCE_AUDIT.md`

`docs/OMEGA_CORR_TOKEN_INTERFACE.md`

`docs/OMEGA_CORR_DATA_AND_COVERAGE.md`

`docs/OMEGA_CORR_MATCHING_RESULTS.md`

`docs/OMEGA_CORR_CONTROL_EXPERIMENTS.md`

`docs/OMEGA_CORR_ACTUAL_SOLVER_RESULTS.md`

`docs/OMEGA_CORR_EFFICIENCY_REPORT.md`

`docs/OMEGA_CORR_REVIEWER_AUDIT.md`

`docs/OMEGA_CORR_FINAL_RESEARCH_REPORT.md`

机器可读结果：

`results/omega_corr_token_coverage.csv`

`results/omega_corr_candidate_scores.csv`

`results/omega_corr_mutual_cycle.csv`

`results/omega_corr_paired_controls.csv`

`results/omega_corr_identity_corrections.json`

`results/omega_corr_solver_decisions.json`

`results/omega_corr_full_tracking_metrics.csv`

`results/omega_corr_runtime.csv`

`results/omega_corr_experiment_manifest.json`

所有指标必须有实际原始结果、哈希、模型来源和可复现命令。

不得手工编造数据。

## 18.1 最终科研报告必须回答

1. Ω 的联合视角特征是否优于单视图特征？
2. 哪些 Ω 层具有可验证的对应价值？
3. Patch 有效覆盖率是多少？
4. 目标本身的局部匹配与背景匹配能否区分？
5. Mutual matching 是否优于普通 ROI pooling？
6. Cycle consistency 是否带来额外真实身份纠正？
7. Ω 与旧 VGGT、DINO 的净纠错差异是多少？
8. 是否存在 Ω 独有且有用的身份纠正？
9. 有多少 GMT 错误根本无法使用同步 Ω 对应？
10. 已知真实修正数、破坏数分别是多少？
11. 有多少变化实际通过了 GTA 和 Hungarian？
12. 真实最终 ID 改变了多少？
13. 不确定性触发的调用预算与系统延迟是多少？
14. 为什么不能直接使用 Native Prior 或普通 DINO？
15. 这项工作距离 WWW 论文要求的独立贡献还有哪些差距？

不能用“成功实现新的网络结构”代替以上回答。

---

# 19. 资源预算

本次优先完成冻结模型机制验证。

建议采用有限执行预算：

第一阶段：

最多约 96–160 组实际图像对，优先复用已经存在的合法固定数据。

第二阶段：

完成冻结 Ω、旧 VGGT、DINO 的匹配对照，不进行新训练。

第三阶段：

只有发现有效独立身份信息，才训练小型分数校准模块。

第四阶段：

只有实际 solver 净纠错成立，才执行完整在线视频。

第一阶段额外研究缓存目标不超过约 2GiB，不包含已存在的原始模型权重。

避免保存所有帧的 24 层 dense tokens。

优先保留选定图像对与候选目标的有限局部特征。

当前已有 V100 研究环境记录，先检查真实 GPU 和任务占用。

不能抢占其他项目资源。

所有模型推理都应记录实际的 CUDA 完整耗时与 peak memory。

不能因为 GPU 吞吐较低就悄悄省略 Ω 开销。

---

# 20. FINAL GOAL 与执行原则

本次最终 Goal 是：

**在冻结 GMT 和 VGGT-Ω 的前提下，证明或者否定 VGGT-Ω 的多视角中间 Tokens 能否为真实跨摄像头目标提供独立、可验证的局部对应证据，并检验这些证据能否合法改变 GMT 原有身份决策、获得完整视频净收益。**

这个 Goal 包含四项研究结论：

- `TOKEN_CORRESPONDENCE_EVIDENCE`
- `UNIQUE_OMEGA_EVIDENCE`
- `NATIVE_GMT_CORRECTION`
- `FULL_TRACKING_EFFICIENCY`

四项必须单独给出状态。

不能出现：

模型能跑通，就宣布身份纠错成功。

匹配循环能闭合，就宣布三维几何正确。

离线 Recall 提升，就宣布 HOTA 提升。

输出 ID 变化，就宣布 GT 身份正确。

也不能把以前 Ω-F 的负实验当成这次 Ω-Corr 的失败证据。

## 现在立即执行

步骤一：检查最新远程分支、源码、官方 Ω512 权重、原始 GMT checkpoint、完整 TRAIN 审计和已有固定96图像对；创建隔离研究分支。

步骤二：实现真实 Ω 多视角中间 Token 提取，核对四层形状、2048维分半、17个特殊 token、patch16 和真实检测框坐标。

步骤三：建立目标局部 patch 及有效覆盖率审计，证明没有复制虚假点特征。

步骤四：在合法 TRAIN 候选上执行 Frozen Omega 单视图/多视图、frame-half/inter-frame-half、MNN、Cycle 和背景对照。

步骤五：完成旧 VGGT、DINO、Native GMT 的同候选强对照。

步骤六：生成真实已知身份纠错/破坏统计，完成场景、人物和不同尺寸分桶。

步骤七：若存在独立可用信息，训练极小的 Correspondence Calibration Head；否则保存预设负结果。

步骤八：若候选净纠错成立，使用原生 GMT state forks 测试实际 GTA、Hungarian、阈值、Bank 和最终 ID。

步骤九：若实际求解有可信净收益，测试不确定性触发 Ω 以及完整视频的 HOTA、AssA、CVIDF1、IDSW 和真实系统速度。

步骤十：完成全部注册独立实验、保存失败和正向结果、回答审稿问题，提交推送最终报告。

**重要：不因第一个模块效果不好就停止所有已注册的独立机制验证；也不允许在没有合理证据的情况下无限扩大实验。**

每个阶段真实完成后立即 Git commit 和 push，并核对远程 HEAD。

最终只接受有真实数字、真实错误案例、完整输入来源和代码可复现的科研结论。

现在开始执行，不要只输出计划。