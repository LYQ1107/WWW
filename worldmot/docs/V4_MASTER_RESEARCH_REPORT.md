# WorldMOT V4 — master research report

本报告严格区分已通过的工程测试与尚未完成的真实实验。R2/R3 的原结果、权重与感知缓存不变。

当前 CPU 测试：**247 项，失败 0 项，跳过 0 项**。
执行代码指纹：`5e2ef23b466094f1edfdd9e690a6e925b83ea5399dc174f47198ddb2214ed0b1`。

GMT 原始完整轨迹已从原正式评测证据定位到 `fulltrain_20261022/test/gmt_native`，仍需恢复后逐文件复核 Manifest/轨迹 SHA。检测候选缓存不会作为 GMT Native Global ID。

| 阶段 | 实际状态 |
|---|---|
| A_GMT_SYNCHRONOUS_IDENTITY_AUDIT | WAITING_FOR_NAS_POWER_RESTORATION |
| B_FROZEN_GEOMETRY_CAUSAL_AUDIT | WAITING_FOR_NAS_POWER_RESTORATION |
| C_WORLD_OBJECT_STATE_IMPLEMENTATION | CPU_PASSED_REAL_VERIFICATION_PENDING |
| D_TRAIN_VAL_STATE_VERIFICATION | WAITING_FOR_NAS_POWER_RESTORATION |

实现包括独立 G1/G2/G3/G4 控制、六维常速度预测、Joseph Kalman 更新、身份/空间分离、置信度门控长短期视觉记忆、保守多视角测量和因果 Ω 请求。
C0 默认路径用冻结 R2 源码对照；CPU 合成测试的 80 时刻完全一致不等价于真实 checkpoint 的全视频复现。
完整 TRAIN 回放必须再次验证 C0 的每个输入张量、决策与数值状态。

冻结身份头的最终 all24 TRAIN 权重已经看过六个 TRAIN VAL 场景。本轮是 TRAIN 内部机制开发，不是独立泛化验证。
正式 TRAIN VAL TrackEval 仅使用原始 GT 合法的 4/8/12；16/20/24 仅作干净标注子集诊断。
默认时间单位为 frame_id tick，空间单位为相对世界坐标；不假定秒或米。

未获得真实指标时，不能回答 V4 是否超过 R2、No-Geometry 或 GMT。CSV 留空未测值，不填零。
缓存请求次数不等于真实 FPS；稀疏请求后的 Live keyframe 对齐和 RGB-to-ID 延迟必须另测。

下一步：NAS 恢复后核验空间与不可变输入，先运行有限真实工程预检，随后完成预注册六场闭环及单步归因。
本轮不启动新的 24 TRAIN 长训，不执行 V4 正式 TEST 跟踪或 TEST 参数选择。

来源与许可详见 [V4_SOURCE_ADAPTATION.md](V4_SOURCE_ADAPTATION.md)；具体固定范围见 [V4_PREREGISTRATION.md](V4_PREREGISTRATION.md)。
