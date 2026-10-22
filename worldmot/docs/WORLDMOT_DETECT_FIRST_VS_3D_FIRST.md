# WorldMOT：先检测再重建 vs 先三维再检测

本实验在 `research/worldmot-detection-reconstruction-ablation-20261022` 隔离分支完成，运行时完全独立于 GMT。基线为 `846a73004a0c1f1514324e8596234614d4644fab`，使用原始 VisionTrack RGB 和原始 GT；GT 只在离线评测阶段读取。

## 实际比较的系统

* **A（detect-first）**：两张完整同步 RGB → MMDetection RTMDet-Ins-tiny COCO person detector（bbox、mask、score）→ 官方 VGGT-Ω 512（同样输入完整原图）→ mask 下采样点图、协方差和双向重投影约束 → 3-D fusion → `WorldIdentityMemory` → 目标视角回投影。
* **B0（已有三维优先对照）**：PanSt3R v0 原生联合几何/实例 → `WorldIdentityMemory`。它不是 VGGT-Ω 3-D query head，结果没有被标成 Omega 结果。
* **B1（Omega 原生 3-D person query）**：接口已经注册，但公开 VGGT-Det checkpoint 的 ScanNet 18 类没有 `person`，本机没有 VisionTrack-person checkpoint，因此状态是 `PERSON_3D_HEAD_REQUIRES_TRAINING`，没有进入数值排名，也没有用随机权重推理。

RTMDet 记录为 MMDetection 3.3.0、MMEngine 0.10.7、MMCV 2.2.0；Ins-tiny checkpoint SHA256 为 `ec670f7ee9e20bd7931e15f15b7016f7fe531baaab81f2e6153382d046111885`。Ω 采用已核验的官方 512 checkpoint，SHA256 为 `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`。四卡只并行不同实验，表中 FPS 始终是单个同步双摄时刻的处理速度。

## 评测协议

Scene 8 和 Scene 9 各使用连续 100 个同步双图，Scene 24 使用拥挤场景连续 20 个同步双图。A/B0 使用相同帧、摄像头顺序、原始 GT、IoU=0.30、逐帧一对一匹配和可见性规则。投影只有在当前目标视角存在检测支持时才计入正式 2-D 评测；LOST 轨迹的纯预测框不会被当成真实观测。AP 在本次单阈值保存结果中记为 `null`，因为没有伪造 score sweep；原始 detector 的 operating-point precision/recall 单独记录。

## 结果

| 场景 | 系统 | 2-D precision / recall | HOTA | IDF1 | AssA | IDSW | 跨视角同 ID | 平均投影 IoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 8 / 100 帧 | A | 0.722 / 1.000 | 0.839 | 0.828 | 0.975 | 10 | 0.995 | 0.817 |
| 8 / 100 帧 | B0 | 0.659 / 0.910 | 0.778 | 0.756 | 0.979 | 6 | 0.982 | 0.520 |
| 9 / 100 帧 | A | 0.899 / 1.000 | 0.860 | 0.847 | 0.823 | 15 | 0.980 | 0.760 |
| 9 / 100 帧 | B0 | 0.410 / 0.768 | 0.488 | 0.430 | 0.653 | 20 | 0.852 | 0.524 |
| 24 / 20 帧 | A | 0.761 / 0.438 | 0.493 | 0.409 | 0.632 | 89 | 0.590 | 0.614 |
| 24 / 20 帧 | B0 | 0.363 / 0.187 | 0.337 | 0.216 | 0.807 | 11 | 0.222 | 0.535 |

Scene 8/9 的 A 路线同时提高召回、precision、投影 IoU、HOTA 和 IDF1；跨视角同 ID 也高于 B0。Scene 24 中 A 的检测和跨视角关联明显好于 B0，但拥挤场景身份交换仍多（89 次），所以这 20 帧只能作为诊断，不能替代完整视频 MOT 结论。A 的 RTMDet 原始检测在 Scene 8/9 的 recall 都是 1.000，precision 分别为 0.701 和 0.899；Scene 24 降到 0.478 recall，失败主要来自遮挡和拥挤检测。

## 速度

端到端包含 RGB 读取、模型、三维处理、世界对齐、身份管理、投影和最终交付；单位是同步双摄时刻/秒。

| 场景 | 系统 | 端到端 mean / P50 / P95 (s) | 端到端 pair FPS | 模型-only mean / pair FPS |
|---|---|---:|---:|---:|
| 8 | A | 1.003 / 0.976 / 1.131 | 0.997 | Ω 0.178 s / 5.62 FPS；detector 0.180 s / 5.57 FPS |
| 9 | A | 0.965 / 0.949 / 1.041 | 1.036 | Ω 0.179 s / 5.58 FPS；detector 0.184 s / 5.43 FPS |
| 24 | A | 1.637 / 1.572 / 1.743 | 0.611 | Ω 0.183 s / 5.46 FPS；detector 0.274 s / 3.65 FPS |
| 8 | B0 | 2.344 / 2.280 / 2.767 | 0.427 | PanSt3R forward 0.812 s / 1.23 FPS |
| 9 | B0 | 2.443 / 2.387 / 2.933 | 0.409 | PanSt3R forward 0.883 s / 1.13 FPS |
| 24 | B0 | 2.584 / 2.578 / 3.178 | 0.387 | PanSt3R forward 0.800 s / 1.25 FPS |

Scene 8 的 A 逐阶段 P50 为：RGB 0.052 s、检测 0.156 s、Ω 0.176 s、检测到三维 0.059 s、世界对齐 0.294 s、身份管理 0.0009 s、回投影 0.0022 s。B0 的主要开销是 PanSt3R geometry 1.337 s 和世界对齐 0.900 s。完整 P50/P95 阶段数据在 `results/worldmot_detection_order_fps.csv`。

## 公平性修复和错误归因

共享 pipeline 已完成四项修复：RTMDet 不再是占位 detector；ORB 静态对应点同时屏蔽当前/历史合法人物 mask；投影输出区分当前观测、纯几何预测、遮挡未知和 LOST；评测器使用原始 GT、固定 IoU 和完整输入序列的 IDF1/AssA/HOTA/CVIDF1 统计。

每个 A detection 保存 bbox、score、mask 的无损 row-major RLE、当前 Ω 点图支持、协方差、融合状态、重投影覆盖、最终 ID 和回投影结果。A 的跨视角融合不是最近质心合并，而是协方差/Mahalanobis、尺度归一化距离和双向 mask/bbox 覆盖门控；无法可靠判断时保留为单视角并记录弃权原因。

失败样本已按 `person_detection`、`geometry_reconstruction`、`cross_view_fusion`、`world_alignment` 和 `projection_or_identity` 分类，见 `results/worldmot_detection_order_failure_cases.json`。同一帧的两路图、检测/GT/回投影框和 3-D 实例见 `results/detection_order/visuals/a_scene_08_frame_000001/`。

## 调度和低频重建注册状态

5 帧执行顺序控制已通过：在同一完整 RGB 输入上交换 detector 与 Ω 的执行先后，检测输出和点图形状/数值一致；该控制不进入 A/B0 排名，记录在 `results/detection_order/schedule_control.json`。逐帧 Ω 是本次正式速度基线。每 5 帧/每 10 帧更新的优化尚未进入正式结果：中间帧若没有当前合法的相机位姿和三维位置推断，系统不会复用旧人物深度，因此这些 cadence 结果标为 `NOT_ADMITTED_NO_CURRENT_3D_INFERENCE`，而不是用陈旧三维位置冒充加速收益。

## 结论

在这批真实连续视频上，**先二维人物检测再用 Ω 做完整双视角几何并融合**是当前应接入 WorldMOT 的主线：Scene 8/9 的准确率和跨视角身份净质量更高，端到端约 0.997–1.036 pair FPS，也快于 B0 的 0.387–0.427 pair FPS。拥挤 Scene 24 证明 A 的检测和去重更强，但持续身份管理仍需单独改进，不能宣称 MOT 已在拥挤场景完全解决。

“Ω 原生三维检测能否减少重复实例”目前没有科学答案：B1 的有效 person query head 尚未获得合法训练权重，PanSt3R B0 也不能冒充 B1。只有完成 VisionTrack person-supervised head 的注册训练并通过相同监督预算、同一连续视频协议后，B1 才能加入正式比较。
