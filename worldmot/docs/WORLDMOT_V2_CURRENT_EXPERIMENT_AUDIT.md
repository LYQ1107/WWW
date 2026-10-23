# WorldMOT v2 当前实验审计

更新时间：2026-10-23 13:11:43 UTC  
分支：`research/worldmot-v2-fulltrain-test-20261022`  
状态：**进行中，不能视为最终结论**

这份报告只记录已经存在的真实产物；未完成的实验明确列出，不补造结果。

## 已完成

### TRAIN 与冻结

- 修正后的 RGB→RTMDet BGR 输入协议已用于正式缓存。
- VisionTrack TRAIN 24/24 场景、28,754 个同步双摄时刻、57,508 张图像已完成对象特征缓存。
- 完整监督/缓存审计通过；缓存和源代码 SHA 已锁定。
- 三个主 v2 seed（20261022/23/24）和四个注册控制均完成 4 个 full-TRAIN epoch，并在独立 TEST 前冻结。
- TEST 没有参与模型或阈值选择；冻结锁状态为 `FROZEN_BEFORE_INDEPENDENT_TEST`。

### 完整 TEST 推理

| 方法 | 完整场景 | 状态 |
|---|---:|---|
| `gmt_native` | 22/22 | 完成 |
| `v1` | 22/22 | 完成 |
| `v2_joint_seed20261022` | 22/22 | 完成 |
| `v2_joint_seed20261023` | 22/22 | 完成 |
| `v2_joint_seed20261024` | 22/22 | 完成 |
| `v2_no_appearance_seed20261022` | 19/22 | 运行中 |
| `v2_no_omega_seed20261022` | 21/22 | 运行中 |
| `v2_two_stage_seed20261022` | 14/22 | 运行中 |
| `v2_no_geometry_seed20261022` | 6/22 | 运行中 |

当前运行中的最新进度：no-appearance Scene 20 为 256/1006；no-Ω Scene 22 已启动但尚未写入 progress；two-stage Scene 15 为 1664/1904；no-geometry Scene 7 为 128/2055。四个进程均仍存活。no-Ω Scene 21 的完整 manifest 已核验为 1052 帧、真实端到端 0.502161 FPS。

## 当前已得到的 v2/GMT 结果

这是三个主 v2 seed 与 GMT 的中途 TrackEval 诊断，不是全部注册方法的最终官方汇总。使用相同 `tracks_2d_by_camera` 输出协议和原始 GT；14 个 GT 合法场景实际进入评分。

| 方法 | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |
|---|---:|---:|---:|---:|---:|---:|
| v2 三 seed 均值 | 37.256 | 45.032 | 31.364 | 44.937 | 37.893 | 944 |
| GMT native | 66.065 | 69.579 | 63.066 | 80.112 | 72.964 | 6044 |

v2 相对 GMT 暂时为：HOTA −28.809、AssA −24.547、DetA −31.702、IDF1 −35.175、MOTA −35.071。v2 的 IDSW 较低，但 FN 明显更多，不能据此宣称身份质量更好。

22 个完整 TEST 视频的真实端到端双摄时刻 FPS：GMT 3.331；v2 三个 seed 分别为 0.810、0.807、0.792，三 seed 合并墙钟为 0.803。

## 尚未完成

- 四个注册 v2 消融尚未完成全部 22 个 TEST 场景。
- 协调器尚未启动最终全方法官方评测、统一指标导出和失败案例汇总。
- 8 个场景（5、12、13、14、17、20、21、22）因原始 GT 重复/版本非法被官方 GT 审计阻断；没有修改 GT，也没有把这些场景伪装成合法分数。
- capacity study 的后续 TEST 仍等待原始注册 TEST 和容量门控完成。
- 因此当前不能给出“全部方法最终排名”或“v2 已成功/已失败”的最终科学声明。

## 当前判断

主线 v2 已显示出相对 GMT 的明显总体差距，因此不应继续追加新的架构或调参来追逐 GMT。正在运行的消融仅用于完成预注册的因果诊断：确认几何、外观和 two-stage 是否分别造成损害，以及判断失败来自组件、检测覆盖还是训练/容量问题。消融不会改变主线 v2 的既有分数，也不用于 TEST 选择。

原始详细结果：`results/worldmot_v2_vs_gmt_midrun_trackeval.json`。  
实时状态：`results/worldmot_v2_independent_test_live_status.json`。
