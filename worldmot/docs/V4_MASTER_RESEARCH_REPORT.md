# WorldMOT V4 — master research report

本报告严格区分已通过的工程测试与尚未完成的真实实验。R2/R3 的原结果、权重与感知缓存不变。

当前 CPU 测试：**279 项，失败 0 项，跳过 0 项**。
执行代码指纹：`f3ab66e050d8e88346a9cd95fe8012a925bfa6b43943392bcef127c7b86aeee0`。

GMT 原始完整轨迹已从原正式评测证据定位到 `fulltrain_20261022/test/gmt_native`。逐文件复核 Manifest/轨迹 SHA 的完成状态以阶段 A 结果为准。检测候选缓存不会作为 GMT Native Global ID。

| 阶段 | 实际状态 |
|---|---|
| A_GMT_SYNCHRONOUS_IDENTITY_AUDIT | STAGING_BYTE_VERIFIED_LOCAL_INPUTS |
| B_FROZEN_GEOMETRY_CAUSAL_AUDIT | STAGING_BYTE_VERIFIED_LOCAL_INPUTS |
| C_WORLD_OBJECT_STATE_IMPLEMENTATION | CPU_PASSED_REAL_VERIFICATION_PENDING |
| D_TRAIN_VAL_STATE_VERIFICATION | STAGING_BYTE_VERIFIED_LOCAL_INPUTS |

实现包括独立 G1/G2/G3/G4 控制、六维常速度预测、Joseph Kalman 更新、身份/空间分离、置信度门控长短期视觉记忆、保守多视角测量和因果 Ω 请求。
C0 默认路径用冻结 R2 源码对照；CPU 合成测试的 80 时刻完全一致不等价于真实 checkpoint 的全视频复现。
完整 TRAIN 回放逐时刻验证 C0 的每个输入张量、决策与数值状态；有限工程预检不能代替完整覆盖。

冻结身份头的最终 all24 TRAIN 权重已经看过六个 TRAIN VAL 场景。本轮是 TRAIN 内部机制开发，不是独立泛化验证。
正式 TRAIN VAL TrackEval 仅使用原始 GT 合法的 4/8/12；16/20/24 仅作干净标注子集诊断。
默认时间单位为 frame_id tick，空间单位为相对世界坐标；不假定秒或米。

未获得真实指标时，不能回答 V4 是否超过 R2、No-Geometry 或 GMT。CSV 留空未测值，不填零。
缓存请求次数不等于真实 FPS；稀疏请求后的 Live keyframe 对齐和 RGB-to-ID 延迟必须另测。

剩余流程以阶段状态为准：预注册六场闭环及单步归因、原始合法 TRAIN 场景评分、真实有限 RGB-to-ID 预检。输入存储故障和不完整尝试均需独立记录。
本轮不启动新的 24 TRAIN 长训，不执行 V4 正式 TEST 跟踪或 TEST 参数选择。

来源与许可详见 [V4_SOURCE_ADAPTATION.md](V4_SOURCE_ADAPTATION.md)；具体固定范围见 [V4_PREREGISTRATION.md](V4_PREREGISTRATION.md)。

恢复过程：第一次 NAS 输出尝试的两段真实工程预检共 32/32 时刻严格复现 C0；随后六场回放、原 TEST 输出审计及保留案例进程退出，未形成完整视频分数。同一期间观察到 NAS I/O 错误，可读取日志已保存，但缺少逐进程异常栈。详见 `results/v4_execution_attempts.json`。当前准备原 SHA 校验的本地输入镜像与本地输出，模型参数和评测范围不变。
