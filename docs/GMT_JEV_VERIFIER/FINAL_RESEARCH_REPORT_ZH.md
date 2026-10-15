# GMT-JEV V2 中文研究报告

当前科学判断：**BLOCKED_INSUFFICIENT_SUPERVISION**；SCIENTIFIC_SUCCESS=false。

A. 分支 research/gmt-jev-typed-verifier-20261015；当前Goal状态 BLOCKED_INSUFFICIENT_SUPERVISION。准确发布HEAD以runtime/publication/LAST_VERIFIED.json为准；本报告生成HEAD为 f0636b65f51575f8d6c070da517a947da37f0099，不冒充未来发布提交。

B. 完整GMT与原MLP step300文件/源码SHA冻结。P0原GMT完整TRAIN记录逐项复核见各case；原MLP旧严格证据复用，未声称运行了新Typed OFF全视频合同。

C. Q1使用既有已提交历史，输出RELIABLE/CONFLICTED/UNKNOWN；Q2拟在同状态下选择完整KEEP/APPLY计划。P0未准入时只做Q1诊断，没有Q2训练或新部署策略。原生配对克隆各执行一次真实提交用于诊断，主轨迹仍只提交一次；不等于已实现学习式JEV控制。

D. 真实监督支持：

| Q2动作（非时间留出） | 认证完整计划 | 视频内人物连接依赖组 | video-person组 | 视频 |
|---|---:|---:|---:|---|
| APPLY_MLP | 8 | 7 | 8 | [12, 13, 14] |
| KEEP_GMT | 0 | 0 | 0 | [] |

| Q1类别 | 过去候选身份遭遇次数（有依赖） | 不同video/nativeID组 |
|---|---:|---:|
| RELIABLE | 37054 | 45 |
| CONFLICTED | 11382 | 19 |
| UNKNOWN | 54508 | 367 |

UNKNOWN计划比例：全部 1.1746%；MATCH分歧配对样本中 91.7526%。完整逐行四象限、候选缺失、别名和Q2梯度诊断见机器记录。

Q1诊断（不参与在线身份提交）：

| Q1视频14诊断类 | Precision | Recall | F1 | 真值样本 |
|---|---:|---:|---:|---:|
| RELIABLE | 12.9032% | 2.5000% | 4.1885% | 320 |
| CONFLICTED | 24.0076% | 88.1944% | 37.7415% | 144 |
| UNKNOWN | 92.2830% | 76.6355% | 83.7345% | 749 |

预设softmax≥0.9的CONFLICTED预测数：0；固定外观代理预测数：0。没有按结果调低阈值。
Precision/Recall按采样诊断行计算，不是独立人物统计显著性或已校准置信度。历史分类不等于在线保护；该头从未参与native commit。

E. 同条件冻结历史成绩：

| 冻结历史参考 | HOTA | AssA | DetA | IDF1 | CVIDF1 | CVMA | MOTA | IDSW | Frag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GMT | 68.054089 | 67.064493 | 69.411013 | 87.182280 | 86.544619 | 87.093920 | 87.000434 | 1898 | 503 |
| GMT+MLP | 68.654661 | 68.098573 | 69.552818 | 88.649302 | 88.001081 | 87.154018 | 86.928650 | 1945 | 501 |

MLP相对GMT：HOTA+0.600573、AssA+1.034080、CVIDF1+1.456462、IDSW+47、Frag−2，MOTA约−0.0718。这些均是旧开发结果，不归给新Q1/Typed。

F. 相对现有MLP，本轮尚未建立新的在线收益。Q1诊断不改变任何提交；无法给出新HOTA/IDSW改善值。

G. Typed相对普通验证器的独立贡献NOT_ESTABLISHED；Q2未准入时二者均不训练，没有用弱对照包装胜出。

**JEV_VERIFIER_INDEPENDENT_VALUE_NOT_ESTABLISHED**。

H. 历史完整图像FPS为GMT3.409768、MLP3.129765、缓存MLP2.848982。新Verifier FPS尚未测；不将P0审计或CPU小头速度冒充完整FPS。

I. TRAIN16已暴露；Q1 TRAIN14诊断也不属于合格独立视频。独立完整视频与必要训练种子仍PENDING。

J. 现有证据尚不足以支撑双问题Typed-JEV的WWW创新主张；最缺可靠的两动作监督、真实结构/机制增量和独立任务验证。详细R1–R10见 REVIEWER_RISK_ANALYSIS.md。

本轮资源：{"created_Asia_Shanghai": "2026-10-15T15:17:07.653476+08:00", "new_GPU_hours_including_live": 1.1772130608724223, "new_GPU_hours_max": 4, "old_GPU_hours_unchanged": 11.92904806540865, "new_retained_MiB": 1.4122848510742188, "home_free_GiB": 50.00343704223633, "registered_live_jobs": []}

原生局部分支证据见 NATIVE_PAIRED_PLAN_OPPORTUNITY.json；图像/标注/配置/权重字节复核见 INPUT_PROVENANCE.json；实际设备与环境见 ENVIRONMENT_PROVENANCE.json。正常、提交前异常及真实提交后异常的状态/RNG恢复记录在 CONTRACT_TESTS.json。

失败处理：若全部授权TRAIN与既有日志仍无足够KEEP/APPLY监督，按任务书停止本轮结构探索，保留GMT+MLP。不会因还有GPU预算而增加Question、制造标签或改用封存数据。

最终路线：**BLOCKED_INSUFFICIENT_SUPERVISION**。原MLP继续作为最强已测开发参考；科学目标未完成。

两次短工程预检的主动中止、原因与日志均保留，并计入1.177 GPU-hours；它们不作为完整科学结果。后续完整审计与固定终点Q1诊断未改变标签、准入门槛或根据TRAIN16选模。
