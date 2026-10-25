# 身份特征、坐标和缓存契约

## 真实身份特征

`WorldObjectObservation` 统一携带：camera_id、detection_id、当前 bbox_xywh、score、
roi_feature、feature_valid、world_center、world_covariance、spatial_support、geometry_valid、
geometry_reliability、timestamp、geometry_status、visible、coordinate_frame 和 source。
底层 `PersonDetection.view_id` 在适配器中映射为 camera_id，并验证来源一致。

维度 D 来自 `DETECTOR_LOCK.json:roi_feature_dim`；原始 NPZ 的 `roi_features.shape[1]`
必须一致。本轮不读取真实候选缓存，因此不报告一个未经本轮检查的真实 D。
合成测试分别使用 D=3 和 D=8，不能把它们当成 GMT 模型维度。

特征在原始维度内做 L2 归一化。多视角融合权重为
`score * (0.5 + 0.5 * geometry_reliability)`；完整无几何模式仅使用 score。
仅当前可见、特征合法、权重大于零的视角参与。缺失、零向量、非有限或维度不符都会显式报错；
`GEOMETRY_ONLY_ABLATION_V3` 是单独的方法，不能冒充完整视觉身份模型。

历史身份与空间分别更新。实际分配成立后，score≥0.5、视觉相容、没有严重空间冲突的观测
才能更新身份。候选为 `normalize(current_ROI + 0.1*tanh(memory_delta))`；历史通过归一化 EMA
更新，步长为 `(1-memory_momentum)*update_gate`。未匹配或不可见对象不写入身份证据。
每个视角最多记录4个近期真实 ROI，不能用 camera_id、GT embedding 或 GMT 最终 Global ID 代替特征。

## 三种坐标

| 名称 | 含义 | 合法转换 |
|---|---|---|
| `OMEGA_LOCAL_WORLD` | 当前同步双图自身的重建世界坐标 | 相机位姿投影 / 当前有效 Sim(3) |
| Camera Frame | 当前摄像头坐标，+z 为前向深度 | `camera_from_world = inverse(world_from_camera)` |
| `PERSISTENT_WORLD` | 当前场景持续对象的公共历史坐标 | 当前时间的已验证 `WorldFrameTransform` |

Ω 解码 extrinsic 是 Camera-from-Ω-Local-World。`_unproject_depth()` 使用其逆变换生成
公共局部点图；`CameraModel.world_from_camera` 保存逆 extrinsic。
深度聚类使用 `camera_from_world` 的第三行与三维点做乘积，或严格对齐像素的 Depth Head。
不再使用世界 z 或点的范数作为相机前向深度。

`WorldFrameTransform` 实现 `p_persistent = s R p_local + t`，协方差为
`Σ_persistent = s² R Σ_local Rᵀ`。尺度必须为正，旋转正交且 det=+1；变换无效或时间不匹配会拒绝写入。

RGB box 使用 Ω `PixelMapping.matrix` 的 Crop→Resize→Pad 和半像素修正进入模型点图坐标。
四角一起变换、裁剪；缺失变换不默认为 identity。无效深度、相机后方点、非有限坐标被屏蔽；
投影越界、无有限支持、深度不可恢复等返回显式状态和零几何可靠性。
深度簇仍是 box 内的前景提案，不被称为真实实例 mask。

## v1 只读联结

`FrozenGMTFeatureJoin` 按 `(scene, timestamp, camera_id, detection_id)` 联结旧支持缓存与原始候选缓存。
验证冻结 detector-lock SHA、原始 shard SHA、候选数量、键的一一对应、bbox、score、image_ids，
以及已保存时的 RGB SHA/候选 shard SHA。ROI 维度与 Manifest 一致才返回。
旧 v1 sidecar 没有记录独立 SHA，此历史限制不能用当前代码补成“当时已验证”。

v1 的身份字段在推理 reader 中被隔离，不向 Observation 暴露。没有联结来源时，要求视觉特征的
路线直接报告 `FEATURE_UNAVAILABLE`。v1 的几何一律标为 `LEGACY_CAMERA_DEPTH_UNVERIFIED`，
完整几何路线拒绝将它无损升级。

## v2 新 Schema

`worldobject_omega512_sparse_support_v2` 在独立目录生成，禁止覆盖已有 v1/COMPLETE Manifest。
新字段包括原维 ROI、feature_valid、geometry_valid、geometry_status、当前相机模型、
image_to_model、当前对齐证据、RGB SHA、原始候选 shard SHA、源 commit/代码 SHA，
并为 NPZ 和 JSON sidecar 分别登记 SHA。感知缓存不保存 identity targets。

支持数组仍包含 box/foreground 点集、计数、中心、协方差、质量和歧义；点明确属于持续世界坐标。
reader 验证时间严格递增、场景完整、候选键唯一；TEST 观测适配器拒绝身份/目标字段。

### 批准后需要重算的最小集合

| 字段 | 能否从旧冻结资产只读恢复 | 后续动作 |
|---|---|---|
| 原始 box、score、candidate key、RGB/image IDs | 能 | 只读校验联结 |
| GMT ROI、D、detector 权重/配置 SHA | 能，从原始 GMT 缓存 | 不需要重跑 GMT/Ω 来补 ROI |
| 旧选中人体点、中心、协方差 | 有，但深度选择可能错误 | 保留为旧结果，不能当成修复后的几何 |
| 正确 Camera-Forward Depth、完整局部点图与相机位姿 | v1 sparse producer 未保存足够信息 | 若另有同源密集资产则校验复用，否则批准后重跑 Ω |
| 重新选择的前景点、中心/协方差、可靠性、世界对齐 | 不能从错误抽样点无损恢复 | 在独立 v2 目录重新生成 |
| TRAIN 身份 targets | 不进入感知缓存 | TRAIN 目标构造器离线匹配，重复身份显式 mask |

本轮只生成了 CPU 合成测试中的微型临时 fixture，没有迁移真实缓存。
v2 目前不支持从单个末尾帧恢复整个场景的世界 gauge；半成品恢复会显式报错，需新目录重建，
不能沿用旧代码“重跑最后一帧即恢复因果状态”的假设。
