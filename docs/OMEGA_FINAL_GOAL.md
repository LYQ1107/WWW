# FINAL GOAL：验证 VGGT-Ω 能否为 GMT 提供独立的跨摄像头几何身份关联收益

## 0. 你的任务是什么

你现在负责 `LYQ1107/WWW` 项目的一轮全新验证实验。

研究对象：

**VGGT-Ω for Geometry-Guided Global Multi-Camera Tracking**

本次不要求你提出新的论文架构，也不允许你随意修改 GMT。

你需要严格按照下面的设计，验证：

**VGGT-Ω 是否能够提供比旧 VGGT 更可靠的跨摄像头几何关系，以及这些几何关系能否真正帮助 GMT 减少身份关联错误。**

请记住：

我们不是要让 VGGT-Ω 取代 GMT。

我们要让：

- GMT 负责检测、ReID、历史关联和全局身份分配。
- VGGT-Ω 负责预测多摄像头空间关系。
- 新增轻量模块负责将可信空间关系转换成候选关联证据。
- GMT 原有 GTA/Hungarian 负责最终身份分配。

整个实验必须按照以下顺序：

**真实几何预测 → 几何可靠性检查 → 真实身份候选区分 → GMT 在线关联验证 → 计算开销评价。**

只有前一个环节真正产生证据，才能声称下一环节取得科学成功。

工程模块能够执行，不代表几何信息有效。

---

# 1. 项目与分支

GitHub：

https://github.com/LYQ1107/WWW

最近的研究分支：

`research/trackvggt-identity-query-www27-20261015`

已经完成的参考提交：

`c6999a7ab1d22549816b5219734f53c8dcb74178`

原始 VGGT 几何研究分支：

`research/vggt-geo-stream-mcmt-www27-20261014`

本次新建隔离研究分支：

`research/vggt-omega-geometry-proof-20261016`

开始前检查当前 GitHub 的远程 HEAD。

如果参考分支已有更新，先读取差异再决定继承内容。

不要覆盖或删除已有研究。

不要重启已经完成的实验。

不要复用旧 JEV 方法。

建议使用独立 Git worktree，防止当前研究与以前的训练或审计相互影响。

## 1.1 必须先阅读的历史证据

阅读以下文件：

`docs/GEO_STREAM_FINAL_REPORT.md`

`docs/GEO_STREAM_GEOMETRY_FAILURES.md`

`docs/GEO_STREAM_REPRODUCIBILITY.md`

`docs/TRACKVGGT_FINAL_REPORT.md`

`docs/TRACKVGGT_FULL_TRAIN_AUDIT.md`

`docs/TRACKVGGT_CORRECTABLE_ERROR_ANALYSIS.md`

`docs/TRACKVGGT_DEV12_NATIVE_ASSET_HANDOFF.md`

`docs/TRACKVGGT_REQUIREMENT_AUDIT.md`

重点理解：

1. 旧 VGGT 几何门控为什么基本没有通过。
2. 旧 B5/B7 为什么没有最终输出差别。
3. 当前 GMT 在 TRAIN 中有哪些真实身份关联错误。
4. 原有候选 Top-8 的覆盖率。
5. 哪些错误有正确且身份纯净的跨摄像头历史候选。
6. 为什么原 AG/QD 不能证明额外身份收益。

已有 TRAIN 审计包含：

- 12 条完整关联训练序列。
- 169,468 次实际 GMT 主关联查询。
- 1,553 次已认证主关联错误或漏关联。
- 大量涉及不可靠或混合历史的事件。

这些审计已经完成，不允许重新从头执行全部 GMT 审计。

优先复用已有原始输入、关联历史、错误索引和哈希。

---

# 2. VGGT-Ω 官方模型与代码

官方 GitHub：

https://github.com/facebookresearch/vggt-omega

论文：

https://arxiv.org/abs/2605.15195

参考说明：

https://github.com/facebookresearch/vggt-omega/blob/main/reproduction.md

必须实际阅读以下源文件：

`vggt_omega/models/vggt_omega.py`

`vggt_omega/utils/pose_enc.py`

`vggt_omega/utils/load_fn.py`

`vggt_omega/models/heads/dense_head.py`

不要根据旧 VGGT 的 API 自己猜测 VGGT-Ω 接口。

## 2.1 模型权重

优先权重：

`VGGT-Omega-1B-416-Reproduction`

对应文件名称：

`vggt_omega_1b_416_reproduce.pt`

优先采用官方 416 复现配置，因为它是官方建议用于基准研究比较的权重。

另一权重：

`VGGT-Omega-1B-512`

只作为单独的自然场景诊断对照。

416 和 512 的结果、预处理、运行成本必须分别记录。

不允许混用权重和输入配置。

VGGT-Ω 权重需要合法 Hugging Face 访问授权。

如果当前服务器没有授权：

1. 检查已有合法缓存和已配置的访问凭据。
2. 不得绕过 Hugging Face 访问控制。
3. 不得使用第三方来源不明的权重冒充官方模型。
4. 记录 `BLOCKED_WEIGHT_ACCESS`。
5. 可以先完成独立源码审计、坐标测试和数据准备。
6. 没有真实模型推理时，不得宣布 VGGT-Ω 科学实验完成。

禁止从头训练 VGGT-Ω。

## 2.2 必须使用的真实接口

官方基本接口：

```python
import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

model = VGGTOmega().to("cuda").eval()
model.load_state_dict(
    torch.load(checkpoint_path, map_location="cpu"),
    strict=True,
)

images = load_and_preprocess_images(
    image_paths,
    image_resolution=416,
    mode="balanced",
).to("cuda")

with torch.inference_mode():
    predictions = model(images)

extrinsics, intrinsics = encoding_to_camera(
    predictions["pose_enc"],
    predictions["images"].shape[-2:],
)

depth = predictions["depth"]
depth_conf = predictions["depth_conf"]
```

必须核验：

- 图像输入的实际 H/W。
- pose_enc 实际形状。
- extrinsics 实际形状。
- intrinsics 实际形状。
- depth 实际形状。
- depth_conf 实际形状。
- 输出是否全部有限。
- 相机旋转矩阵是否合理。
- 模型实际 GPU 显存与耗时。

`encoding_to_camera()` 返回的外参是 camera-from-world，不是 camera-to-world。

不能搞反矩阵方向。

官方 `depth_conf` 为大于 1 的正值形式，不是天然 0 到 1 的概率。

不得对其擅自使用 sigmoid 并称之为官方置信度。

需要独立验证置信度与真实几何残差的关系，才能把它用于几何门控。

---

# 3. 先固定 GMT：不改主干

以下模块不允许替换：

- GMT Backbone。
- GMT Detection Head。
- GMT ReID 特征。
- GMT RPCE。
- GMT GTA / Association Transformer。
- Hungarian。
- 原始 ID 分配逻辑。
- 原始轨迹历史状态。

本次仅替换旧的几何证据来源：

旧：

`VGGT → camera geometry → association residual`

新：

`VGGT-Ω → camera geometry/depth → reliability check → association residual`

涉及的原有代码：

`gtr/geometry/runtime.py`

`gtr/geometry/camera_geometry.py`

`gtr/geometry/geometry_alignment.py`

`gtr/geometry/geometry_confidence.py`

`gtr/geometry/pair_score_adapter.py`

`gtr/modeling/meta_arch/gtr_rcnn.py`

其中：

`GTRRCNN.get_asso()`

是原有关联分数的接入位置。

不要修改原有 Transformer 的参数或结构。

不要重写原有 Hungarian。

不得让新的 Ω 模块改变检测框、检测分数或检测顺序。

## 3.1 新增最小独立适配器

建议建立：

`gtr/geometry/omega_adapter.py`

负责：

- 加载冻结 VGGT-Ω。
- 原始图像预处理。
- 双摄像头联合推理。
- 提取相机内外参。
- 提取深度和有效性信息。
- 返回预处理映射。
- 捕获模型运行异常。

建议建立：

`gtr/geometry/omega_geometry.py`

负责：

- 计算基础矩阵。
- 验证相机外参。
- 验证背景对应。
- 生成目标几何证据。
- 给出可靠性状态与失败原因。

新增配置：

`GEOMETRY_METHOD = "omega_camera"`

默认必须关闭。

不能让旧 `FEATURE_BACKBONE='vggt'` 成为调用 Ω 几何的必要条件。

本次允许：

`FEATURE_WEIGHT=0`

仅运行几何分支。

如果 Ω 关闭，必须逐框、逐 ID 恢复原 GMT 输出。

---

# 4. Phase 0：数据、环境、权重与坐标审计

本阶段先不进行身份关联实验。

## 4.1 检查实际硬件

执行 GPU、CUDA、PyTorch、驱动、显存和正在运行任务的检查。

此前项目报告使用 V100 32GB，不是最初规划的 A100。

所以必须以当前实际机器为准。

不能假设四张 A100 一定可用。

不能抢占其他研究进程。

不要修改当前全局 PyTorch 环境。

如需安装 VGGT-Ω，创建或复用隔离环境。

先单 GPU 跑通真实两张图片。

对于 V100 使用实际支持的精度，例如 FP16；对于 A100 可以验证 BF16。

不得让 Codex 为了解决数值问题擅自换成无来源的量化模型。

## 4.2 坐标映射必须准确

旧 VGGT 使用的 518/14 patch 映射不能直接套用。

VGGT-Ω 的官方预处理可能包含：

- 原图中心裁剪。
- 按长宽比缩放。
- 缩放至 16 的整数倍。
- 不同大小输入间的居中 padding。

需要对每张图保存：

`original_width`

`original_height`

`crop_left`

`crop_top`

`crop_width`

`crop_height`

`resize_width`

`resize_height`

`pad_left`

`pad_top`

`network_width`

`network_height`

建立从原始坐标到网络输入坐标的准确变换矩阵：

`A_original_to_omega`

并实现逆映射，但被中心裁剪丢弃的区域不得假装仍具有有效模型预测。

必须通过人工构造的坐标测试和真实图像框测试。

## 4.3 同一联合推理坐标系

两个摄像头必须在同一次模型前向中处理：

`[Camera A, Camera B]`

不能先独立对 A 推理，再独立对 B 推理，最后直接比较它们各自的世界坐标。

不同独立 forward 的三维坐标系没有自动统一保证。

本轮首先只研究同一时刻的 A/B 相机关系。

跨时间几何传播必须作为后续独立问题，不能偷偷复用旧 F 作为最新几何。

## 4.4 相机顺序验证

对同一对图像运行：

`[A, B]`

以及：

`[B, A]`

将第二种输出按物理 camera_id 恢复。

比较：

- 相对相机姿态。
- 基本矩阵 F。
- 独立背景极线残差。
- 经过正确坐标还原的几何候选分数。

两个方向的 F 必须满足正确的转置关系，容许整体尺度/符号等价。

不能直接要求相机绝对外参的数字逐项相同，因为模型的整体坐标规范可能不同。

还必须排除原 GMT 顺序处理带来的差异。

相机顺序验证只针对 Ω 几何模块，GMT 主跟踪顺序保持既有规范。

完成后提交：

`docs/OMEGA_PHASE0_INTERFACE_AUDIT.md`

---

# 5. Phase 1：建立固定的真实相机对审计集合

这一阶段不要加载 VGGT-Ω 计算所有视频帧。

利用已经完成的 TRAIN 审计，建立固定的两摄像头图像对。

## 5.1 数据来源

使用 VisionTrack 原始 TRAIN 中已有合法数据。

主训练研究场景优先使用：

`[4,5,6,7,8,9,13,17,20,21,24]`

Scene 12 已经用于此前研究的内部验证和已知错误分析，应单独保留为已接触标签的诊断场景，不要当成完全盲测。

严格 DEV5：

`[1,10,11,18,19]`

暂不用于阈值调参或模型选择。

官方 TEST 禁止用于本次设计选择。

## 5.2 首批固定审计样本

第一轮建议不超过 24 组同步图像。

必须覆盖多个场景。

验证模型能否：

- 合法加载。
- 恢复相机参数。
- 输出可用 F。
- 输出深度。
- 完成正确坐标转换。
- 获得独立背景几何验证。

如果这一阶段通过，再扩大为不超过约 100–120 组同步图像对。

建议扩大集合包含：

A. 不依赖 GT 错误选择的均匀时间采样。

B. 不同背景与视角变化的样本。

C. 来自关联 TRAIN 的真实困难关联事件附近的同步图像。

D. 目标尺度不同的样本。

E. 视野明显不足或缺少可匹配背景的负面样本。

建议 11 个 fit 场景各固定最多 6 个均匀时刻，另加入最多 30 个来自合法 TRAIN 的困难事件对应同步时刻；其余额度用于固定诊断。实际样本不足时报告真实数目，不要凑数。

同一人的连续困难帧应做时间去重和分组，防止把高度相关的样本当成独立实验。

必须保留无重叠和失败样本。

不能只报告几何有效的图像。

## 5.3 不允许数据泄漏

选择困难事件可以使用 TRAIN GT 做离线分析，但必须明确标记其为训练诊断，而非独立泛化验证。

模型实际输入不能包含 GT ID。

开发场景图像选择必须在查看新 Ω 的评价指标之前固定。

全部抽样索引、图像路径、物理摄像头 ID 和 SHA 保存到：

`manifests/omega_fixed_pair_cohort_v1.json`

提交并推送。

---

# 6. Phase 2：VGGT-Ω 相机几何是否真的可靠

本阶段是此次研究最关键的第一道科学门槛。

## 6.1 计算基本矩阵

VGGT-Ω 输出：

`E_A = [R_A | t_A]`

`E_B = [R_B | t_B]`

均为 world-to-camera 外参。

定义：

`R_BA = R_B @ R_A.T`

`t_BA = t_B - R_BA @ t_A`

`F_BA = inv(K_B).T @ skew(t_BA) @ R_BA @ inv(K_A)`

其中 K 对应 VGGT-Ω 网络输入图像坐标。

验证数值有限、相机旋转有效、相对平移非退化。

如果视角几乎纯旋转、基线不可靠或矩阵退化，必须标记 INVALID。

将网络图像坐标的 F 正确变换到原始图像坐标：

`F_original = A_B.T @ F_network @ A_A`

这里 A_A/A_B 是原图到对应网络输入的坐标变换。

变换后进行尺度归一化。

必须通过人工相机投影合成测试，检查极线残差、视角顺序、转置和原图坐标映射。

不要只判断矩阵存在就视为成功。

## 6.2 独立背景匹配验证

使用已有背景特征匹配实现：

`gtr/geometry/geometry_confidence.py`

优先检查 ORB/SIFT 等现有模块。

正确屏蔽：

- 检测目标区域。
- 无效 padding。
- 被裁剪后不存在的区域。

获得可靠背景匹配点。

这些匹配点不允许来自 GT 身份关联。

需要：

- 有效匹配数。
- 有效匹配覆盖范围。
- Ω-F 下的对称极线残差。
- 内点比例。
- 残差分位数。
- 失败原因。

重要：

不要在同一批点上拟合传统 F 后，又在完全相同的点上报告无偏精度。

传统 RANSAC 基线应明确区分拟合点与评估点。

若背景匹配足够，可以使用固定空间分块的训练/评估划分，对传统 F 和 Ω-F 使用共同评估点。

不能把传统 RANSAC 的自拟合残差与 Ω-F 的独立残差直接当作公平比较。

## 6.3 和旧 VGGT 对比

必须对完全相同的物理摄像头图像对运行：

- 原始 VGGT 几何。
- VGGT-Ω 416。
- 传统背景匹配得到的相机几何参考。
- 如果 512 官方权重已合法获得，可增加单独的 VGGT-Ω 512 诊断。

比较：

- 有效几何对数量。
- 通过率。
- 背景极线误差。
- 相机顺序稳定性。
- 退化比例。
- 不同场景覆盖率。
- GPU 时间。
- GPU 内存。

注意，原始 VGGT 与 Ω 输入分辨率不同，必须记录各自的实际输入尺寸和有效背景采样，不能把差异全归因于架构。

## 6.4 固定初始有效性参考门槛

可以将旧研究的有效性条件作为第一套固定参考：

- 足够的独立背景点，例如至少 20 对。
- 内点比例至少 0.25。
- 原始像素残差中位数不超过 12px。
- 背景匹配覆盖范围至少达到图像区域的 1%。

其中阈值需要依照已有真实定义使用，不能把原图像素与缩放图像像素混淆。

还应单独报告尺寸归一化的残差，以便比较不同分辨率场景。

这只是预设参考，不代表满足它就一定能准确判断目标身份。

允许画阈值敏感性诊断，但不能因为 Ω 没通过就任意降低门槛，随后把新结果称为主实验成功。

## 6.5 如果又是 0 个有效几何状态

不要直接结束所有分析。

必须分清具体失败原因：

- 权重或数值错误。
- 相机坐标方向错误。
- 预处理坐标映射错误。
- 没有足够独立背景匹配。
- 两摄像头几乎没有重叠。
- 姿态退化。
- Ω-F 与背景匹配不一致。
- 相机顺序变化过大。
- 评估阈值与真实图像尺度不合适。

对独立原因完成有限修复和必要的单元测试。

如果工程错误修复后依然没有有效几何，就记录：

`OMEGA_POSE_NO_RELIABLE_SUPPORT`

不允许通过删除失败样本制造通过率。

这时应完成正式的几何负结果总结，不继续无意义的在线融合。

---

# 7. Phase 3：深度是否能增强跨视角目标对应

只有相机几何通过基础检查后，才研究这个阶段的身份用途。

必须先明确：

**VGGT-Ω 深度图不能直接等价为目标身份。**

两个行人的深度相近，也不能证明是同一个人。

## 7.1 先做 F-only

先实现不使用深度的几何候选分数。

输入：

- 当前检测框。
- 同步对侧摄像头检测框。
- 已验证的 F。
- 几何可靠性。
- 两个相机物理 ID。

输出：

`geometry_score[i,j]`

候选几何分数可从极线距离及目标框尺寸不确定性推导。

可以复用已有：

`soft_box_evidence()`

但必须审计它的数学假设。

因为不同画面的目标框中心通常不是相同三维物理点，所以这种分数只能是弱几何证据。

不要将目标框中心极线残差称为真实的三维定位误差。

## 7.2 再做 depth-assisted geometry

如果 F-only 的背景验证可靠，再检查深度是否能提供额外信息。

可选方式：

- 提取目标脚底附近的候选接地点。
- 在几何支持可靠时获取该区域深度。
- 使用当前联合推理中的相机坐标关系进行三维反投影。
- 再投影到对侧摄像头。
- 比较候选区域是否空间兼容。

但必须遵守：

1. 接地点是近似估计，不是 GT 三维位置。
2. 不能假设每个场景都是平整地面。
3. 遮挡、远距离小目标、脚部不可见时不得强制使用。
4. 不同独立推理的三维坐标不能直接合并。
5. 深度置信度必须经过真实残差校准。
6. 原图没有覆盖的区域不能插值出伪精确坐标。

先以深度有效性作为诊断，不要直接改 GMT 身份。

## 7.3 必须完成的消融

G0：没有几何。

G1：旧 VGGT 姿态 F。

G2：VGGT-Ω 姿态 F。

G3：VGGT-Ω 姿态 F + 可靠性门控。

G4：VGGT-Ω 姿态 F + 有效深度辅助。

G5：几何对照，如固定错误相机配对或打乱的几何关系。

G5 必须采用合法且清晰的安慰剂构造，避免用错误时间或错误相机状态污染真实输出。

如果 G3 与 G2 完全一样，说明门控可能没有决策价值。

如果 G4 不比 G3 好，不能声称深度提供额外作用。

---

# 8. Phase 4：几何究竟能否纠正 GMT 的真实身份错误

注意：

**这是判断研究价值的关键，不是三维重建指标。**

必须使用现有的实际 GMT 关联历史，而不是构造 GT 完美历史。

数据参考：

`docs/TRACKVGGT_CORRECTABLE_ERROR_ANALYSIS.md`

`docs/TRACKVGGT_DEV12_NATIVE_ASSET_HANDOFF.md`

## 8.1 先限定适用范围

本次 Ω 几何是从当前同步 A/B 图像计算的。

因此第一版只允许验证：

- 当前时刻两摄像头的相对几何。
- 当前时刻确实存在的跨摄像头对应候选。

不能直接拿它给十几帧以前的目标位置进行极线比较。

如果某个 GMT 真实错误只涉及过去跨时间参考，没有可用于同一时刻比较的目标区域，必须标记：

`GEOMETRY_NOT_APPLICABLE`

不要伪造历史位置，也不要在没有验证的情况下做几何传播。

同时统计：

- 全部真实 GMT 错误数量。
- 几何适用错误数量。
- 有可靠几何支持的错误数量。
- 正确候选被空间约束包含的数量。
- 几何区分不出正确/错误候选的数量。

## 8.2 不能只计算 Recall@1

对每个拥有已知纯净正确候选的真实错误事件，需要记录：

- 原始 GMT 选中的轨迹。
- GT 认证的正确轨迹候选。
- 正确候选是否在 Top-8。
- 候选属于同摄像头还是对侧摄像头。
- Ω 几何是否有效。
- 正确候选几何得分。
- 错误候选几何得分。
- 正确候选与最难负候选的几何 margin。
- 使用 Ω 几何后排序是否改善。
- 是否破坏原本正确的匹配。

特别重要：

正负候选的标签只允许在离线评价时加入。

几何推理本身不能访问 GT。

## 8.3 必须区分三个层次

### 层次 A：纯几何区分

只看几何证据能否把正确候选排在错误候选前。

### 层次 B：关联矩阵变化

将几何残差加入 GMT 原始 association logits。

检查正确候选的相对排名是否变化。

### 层次 C：实际 GMT 求解决策

检查：

- GTA 轨迹聚合。
- Hungarian。
- 阈值接受或拒绝。
- Bank。
- 最终提交的 track ID。

不能将层次 A 的排名改善称为层次 C 的跟踪纠错。

## 8.4 DEV12 已知错误如何处理

已交付完整 DEV12 资产包含：

- 14,557 次主关联查询。
- 26 个已认证错误/漏关联。
- 23 个错误在完整合法候选中具有纯净正确轨迹。
- 21 个在 Top-8 内。
- 2 个正确候选在 Top-8 外。
- 3 个没有可认证的纯净正例。

这份资产可以用于独立的事后几何诊断。

但这些错误的标签已经被研究者查看。

因此：

不能把这 26 个错误用于选择几何阈值，再将同一批数据宣称为未见过的独立验证。

优先使用其他 TRAIN fit 场景探索，固定参数后再对 DEV12 做一次范围明确的检查。

如果 Ω 只能在这 26 个事件中改变一两个候选排名，必须明确是否产生了真实的 GMT 求解改变。

## 8.5 判断有没有独立收益

真正的正向证据至少需要：

- 可靠几何通过真实独立背景验证。
- 正确与错误身份候选存在有效分离。
- 几何修正产生的已知正确决策增量大于已知破坏。
- 效果不能完全由 Native Prior 或原有 GMT 相似度解释。
- 结果不是来自单一人物的一连串相关帧。

如样本不足，直接报告不足，不要强行把方向判为成功。

---

# 9. Phase 5：只在证据通过后接入 GMT 真实在线推理

这一步是条件执行，不能因为工程代码已经写完就自动运行全部完整视频。

## 9.1 接入位置

原有：

`GTRRCNN.get_asso()`

生成：

`original_logits`

新的几何证据在 GMT 原始 proposal 级关联矩阵上提供有界修正。

形式为：

`new_logits = original_logits + lambda_geo * reliability * delta_geo`

默认：

`lambda_geo = 0`

必须严格保证：

`lambda_geo=0` 时实际预测完全恢复 GMT OFF。

`reliability=0` 时必须返回原始 logits，不能让一个错误的数值变动改变 Hungarian 的 tie-breaking。

## 9.2 第一版融合只处理同一时刻跨摄像头候选

明确约束：

- query_camera != reference_camera。
- query_timestamp == reference_timestamp。
- 几何状态 valid。
- 真实检测框坐标合法。
- 候选确实存在于 GMT 当前合法历史集合。

注意：

GMT 当前因果实现按摄像头顺序处理当前组，只有已处理摄像头的当前观测才会进入后续摄像头的参考集合。

不要为了让两个方向都有几何证据，就改变 GMT 原始处理顺序或偷偷写入未来/尚未发布的身份信息。

首先只在原有规范顺序下接入合法的几何证据。

需要双向同步联合身份分配时，应另立受控实验，不属于本次冻结 Ω 验证。

## 9.3 融合参数

先注册少量固定候选：

`lambda_geo ∈ {0, 0.1, 0.3}`

保证单对候选残差绝对值有上界，例如 1.0。

只允许在关联 TRAIN 的合法拟合部分选择。

不能反复用 DEV5 或 TEST 调整 lambda。

报告每个设置：

- 几何合法关联行数。
- 实际非零残差数量。
- GMT 关联排名改变数量。
- GTA/Hungarian 选择改变数量。
- 已知修正/破坏。
- 最终输出 track ID 变化数量。

如果所有残差都没有改变最终决策，就必须直接报告，而不是写成“增强了 GMT”。

## 9.4 在线验收

首先运行固定短前缀，验证：

- GMT OFF parity。
- 零残差 parity。
- 完整候选映射。
- 空候选行为。
- 真实预测可导出。
- 没有未捕获异常。
- 没有 GT 或未来帧进入输入。

如果存在有证据的净正向纠错，再运行完整合法 TRAIN 视频。

锁定方案后可执行一次严格 DEV5 开发评测，并与同协议 GMT、Native Prior 配对比较。

必须报告：

- HOTA。
- AssA。
- CVIDF1。
- IDSW。
- DetA。
- MOTA。
- CVMA。
- 真实几何调用次数。
- 有效几何次数。
- 完整系统 FPS。
- P50/P95 延迟。
- GPU 峰值显存。

DEV5 已被原 GMT 主干训练见过，不能作为完全独立的未见场景证据。

如果未通过此前几何或身份纠错 gate，就不要消耗 GPU 进行没有机制依据的完整视频重放。

---

# 10. 必须进行的对照

如果进入在线实验，至少有：

B0：原始 GMT。

B1：GMT + Native Prior。

B2：GMT + 旧 VGGT 几何。

B3：GMT + VGGT-Ω 416 姿态几何。

B4：B3 + 可靠性控制。

B5：B4 + 深度辅助（仅当独立有效性成立）。

B6：相同调用预算下的基础几何对照。

不得将旧 VGGT 特征增强收益与新的 Ω 几何收益混合归因。

若 B4 优于 B3，必须解释是哪些不可靠空间约束被关闭。

若 B5 优于 B4，必须证明是深度信息而不是参数或调用次数差异。

若 Ω 只提高了背景极线精度，却没有改变任何正确的目标级匹配，不能声称 MCMT 成功。

---

# 11. 计算资源与存储控制

当前服务器硬件必须重新检查。

以前有 V100 32GB 记录，但不要假设始终如此。

初步设置：

- 第一阶段只使用一张空闲 GPU 运行模型接口。
- 小规模几何验证按实际显存组织任务。
- 对照模型可以使用不同独立 GPU，但需要统计各自完整成本。
- 同时最多使用项目获准的四张 GPU。
- 不影响其他研究进程。

不得训练完整 VGGT-Ω。

不得缓存全部 VisionTrack 视频或大规模 dense features。

只保存：

- 两摄像头物理帧索引。
- 模型 SHA。
- 预处理坐标变换。
- 相机内外参。
- 必要的压缩几何验证统计。
- 小规模对应点。
- 每个真实身份事件的候选几何得分。
- 运行日志和原始指标。

初始额外研究缓存建议控制在 2GiB 左右，模型权重和现有原始数据单独计量。

如需扩大，先证明已有实验存在必要的可用空间证据。

不要因为本次实验失败而自动切换到 Pi3X、MapAnything 或其他模型。

本次任务只验证 VGGT-Ω。

---

# 12. 不能犯的错误

以下行为严格禁止：

1. 重新训练 GMT。
2. 重新训练 VGGT-Ω。
3. 使用未经授权的权重。
4. 将两个独立推理的 world 坐标当成统一坐标。
5. 将 camera-to-world 和 world-to-camera 混用。
6. 将深度置信度直接当成身份概率。
7. 忽略中心裁剪、resize 和 padding。
8. 用错误网络尺度下的 K/F 比较原图目标。
9. 将相机背景几何正确等同于目标身份判断正确。
10. 使用 GT 直接选择在线候选。
11. 使用未来帧建立当前因果几何关系。
12. 将跨时间历史位置直接用当前 F 比较。
13. 用空的有效几何集合计算虚假的 100% 准确率。
14. 删除无重叠或几何失败的场景。
15. 通过降低有效性阈值制造增益。
16. 把离线候选排序改善当成最终 Hungarian 修复。
17. 将场景内大量相关错误帧当成独立人物或独立实验。
18. 修改原始 GT。
19. 用官方 TEST 调参。
20. 不记录负结果就继续训练新模块。

---

# 13. 必须新增的代码和报告

建议新增：

`gtr/geometry/omega_adapter.py`

`gtr/geometry/omega_geometry.py`

`geo_tools/omega_preflight.py`

`geo_tools/omega_prepare_pair_cohort.py`

`geo_tools/omega_pose_audit.py`

`geo_tools/omega_identity_probe.py`

`geo_tools/omega_online_ablation.py`

`geo_tools/omega_finalize.py`

实际文件命名可以与现有风格一致，但不得重复实现已经存在的组件。

需要生成：

`docs/OMEGA_FINAL_GOAL.md`

`docs/OMEGA_SOURCE_AUDIT.md`

`docs/OMEGA_PHASE0_INTERFACE_AUDIT.md`

`docs/OMEGA_CAMERA_GEOMETRY_REPORT.md`

`docs/OMEGA_BACKGROUND_VALIDITY_REPORT.md`

`docs/OMEGA_IDENTITY_CORRECTION_REPORT.md`

`docs/OMEGA_ONLINE_TRACKING_REPORT.md`

`docs/OMEGA_FINAL_RESEARCH_REPORT.md`

`results/omega_pair_cohort.json`

`results/omega_camera_pose_comparison.csv`

`results/omega_background_geometry.csv`

`results/omega_identity_candidate_scores.csv`

`results/omega_identity_correction_audit.json`

`results/omega_runtime.csv`

`results/omega_full_tracking_metrics.csv`

`results/omega_experiment_manifest.json`

没有执行到的条件分支，必须写：

`NOT_EXECUTED`

没有测量的值：

`NOT_MEASURED`

不允许写成 0 或 PASS。

---

# 14. 最终结果状态

必须将科研状态分开：

## OMEGA_GEOMETRY_SUCCESS

VGGT-Ω 在真实 VisionTrack 双摄像头图像中建立了可靠、可独立验证的相机几何关系。

要求有实际背景对应残差、覆盖率、失败率和对照模型结果。

## OMEGA_IDENTITY_EVIDENCE_SUCCESS

可靠几何信息能够在固定的真实困难身份候选中区分正确与错误对应，并产生超过已知破坏的净纠错证据。

不能只依据重建质量或候选分数改变判断。

## OMEGA_TRACKING_SUCCESS

将几何证据合法接入 GMT 后，在完整因果视频中获得可复现的 HOTA、AssA、CVIDF1 或身份错误改善，并且有对应真实决策证据。

## OMEGA_EFFICIENCY_SUCCESS

在完整端到端计时下形成有实际价值的精度—效率权衡。

不要求第一轮就达到 25 FPS，但不得谎称实时。

## SCIENTIFIC_NEGATIVE

如果几何可靠性、身份区分或在线跟踪没有通过，就保留负结果。

具体区分：

- `WEIGHT_ACCESS_BLOCKED`
- `INVALID_GEOMETRY_INTERFACE`
- `POSE_NO_RELIABLE_SUPPORT`
- `GEOMETRY_NOT_APPLICABLE`
- `GEOMETRY_NONDISCRIMINATIVE`
- `NO_NET_IDENTITY_GAIN`
- `NO_FULL_TRACKING_GAIN`

不能将这些不同原因全部笼统写成“模型没有效果”。

如果旧 VGGT 几何失败而 VGGT-Ω 成功，也要分清是几何精度提高、有效样本覆盖提高，还是确实产生身份净收益。

---

# 15. GitHub 提交与实验管理

每个阶段完成后必须：

- 保存真实代码。
- 保存执行命令。
- 记录硬件与软件版本。
- 保存数据/权重 SHA。
- 保存实际结果。
- 记录通过和失败案例。
- Git commit。
- Git push。
- 检查远程 HEAD 与本地一致。

不要直到所有实验结束才统一推送。

每次提交应当能够独立恢复对应阶段的结果。

不要上传官方大模型权重、原始数据集、大缓存或私人凭据。

---

# 16. 现在立即执行的顺序

第一步：

从 `c6999a7a` 对应的可靠 TrackVGGT 分支创建新的 Ω 几何验证工作树。

第二步：

检查当前服务器 GPU、原始 GMT checkpoint、VisionTrack 和已有全量审计资产。

第三步：

检查 VGGT-Ω 权重是否能够合法加载。

第四步：

建立 Ω Adapter 和精确预处理坐标映射，完成真实双摄像头图像推理。

第五步：

实现相机相对姿态与 F 的转换，完成交换相机输入顺序和独立几何数值测试。

第六步：

建立固定的真实同步相机对集合，对旧 VGGT 和 VGGT-Ω 进行背景几何可靠性比较。

第七步：

如果存在足够可靠的几何状态，利用真实 GMT 困难身份候选，验证几何是否具有独立判别力。

第八步：

只有几何身份证据成立时，才在 GMT 原来的 `get_asso()` 路径中加入有界几何残差。

第九步：

如果真实决策出现有意义的净改善，再进行完整视频 HOTA/AssA/CVIDF1 和 FPS 对照。

第十步：

整理完整正负证据，生成最终报告并推送 GitHub。

## 唯一 FINAL GOAL

**验证 VGGT-Ω 的相机几何与深度信息，是否能够在移动双摄像头 VisionTrack 中建立可靠空间对应，并为 GMT 的真实跨摄像头身份关联提供独立且可转化为完整视频收益的证据。**

这次先不要开发新的 AG、QD、JEV 或教师蒸馏模型。

不要将它重新解释成增强 ReID 特征的任务。

核心只有：

**VGGT-Ω 提供空间关系，GMT 提供身份特征，两者通过可信几何约束完成候选身份判断。**

先证明空间关系真实有效，再证明身份纠错，最后讨论在线效率。

不要在第一个失败样本后停止整项审计，但也不要在已经证明无法使用的几何条件上无限调参。

完成预注册的有限实验后，如实给出结论。

现在开始真实执行，不要只回复研究计划。