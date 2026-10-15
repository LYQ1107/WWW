# GMT-JEV V2 中文研究报告

当前科学判断：**P0_NATIVE_CLOSURE_RUNNING**；SCIENTIFIC_SUCCESS=false。

A. 分支 research/gmt-jev-typed-verifier-20261015；当前Goal状态 ACTIVE。准确发布HEAD以runtime/publication/LAST_VERIFIED.json为准；本报告生成HEAD为 8c26a8371ae1cad81f8cdba4e9e23c861197a6ee，不冒充未来发布提交。
B. 完整GMT与原MLP step300文件/源码SHA冻结。P0原GMT完整TRAIN记录逐项复核见各case；原MLP旧严格证据复用，未声称运行了新Typed OFF全视频合同。

C. Q1使用既有已提交历史，输出RELIABLE/CONFLICTED/UNKNOWN；Q2拟在同状态下选择完整KEEP/APPLY计划。若P0失败，本轮只做Q1诊断，没有Q2训练或新native决策。

D. 真实监督支持：
| Q2动作（非时间留出） | 认证完整计划 | 人物连接依赖组 | video-person组 | 视频 |
|---|---:|---:|---:|---|
| APPLY_MLP | 1 | 1 | 2 | [14] |
| KEEP_GMT | 0 | 0 | 0 | [] |

| Q1类别 | 过去候选身份遭遇次数（有依赖） | 不同video/nativeID组 |
|---|---:|---:|
| RELIABLE | 37054 | 45 |
| CONFLICTED | 11382 | 19 |
| UNKNOWN | 54508 | 367 |

UNKNOWN计划比例：全部 1.2670%；实际分歧中 98.9691%。完整逐行四象限、候选缺失、别名和Q2梯度诊断见机器记录。

Q1诊断（不参与在线身份提交）：

| Q1视频14诊断类 | Precision | Recall | F1 | 真值样本 |
|---|---:|---:|---:|---:|
| RELIABLE | 12.9032% | 2.5000% | 4.1885% | 320 |
| CONFLICTED | 24.0076% | 88.1944% | 37.7415% | 144 |
| UNKNOWN | 92.2830% | 76.6355% | 83.7345% | 749 |

预设softmax≥0.9的CONFLICTED预测数：0；固定外观代理预测数：0。没有按结果调低阈值。
Precision/Recall按采样诊断行计算，不是独立人物统计显著性或已校准置信度。历史分类不等于在线保护；该头从未参与native commit。

E. 同条件冻结历史成绩：
| 冻结历史参考 | HOTA | AssA | CVIDF1 | MOTA | IDSW | Frag |
|---|---:|---:|---:|---:|---:|---:|
| GMT | 68.054089 | 67.064493 | 86.544619 | 87.000434 | 1898 | 503 |
| GMT+MLP | 68.654661 | 68.098573 | 88.001081 | 86.928650 | 1945 | 501 |

MLP相对GMT：HOTA+0.600573、AssA+1.034080、CVIDF1+1.456462、IDSW+47、Frag−2，MOTA约−0.0718。这些均是旧开发结果，不归给新Q1/Typed。

F. 相对现有MLP，本轮尚未建立新的在线收益。Q1诊断不改变任何提交；无法给出新HOTA/IDSW改善值。

G. Typed相对普通验证器的独立贡献NOT_ESTABLISHED；Q2未准入时二者均不训练，没有用弱对照包装胜出。

H. 历史完整图像FPS为GMT3.409768、MLP3.129765、缓存MLP2.848982。新Verifier FPS尚未测；不将P0审计或CPU小头速度冒充完整FPS。

I. TRAIN16已暴露；Q1 TRAIN14诊断也不属于合格独立视频。独立完整视频与必要训练种子仍PENDING。

J. 现有证据尚不足以支撑双问题Typed-JEV的WWW创新主张；最缺可靠的两动作监督、真实结构/机制增量和独立任务验证。详细R1–R10见 REVIEWER_RISK_ANALYSIS.md。

本轮资源：{"created_Asia_Shanghai": "2026-10-15T14:44:52.660842+08:00", "new_GPU_hours_including_live": 0.7856501350241403, "new_GPU_hours_max": 4, "old_GPU_hours_unchanged": 11.92904806540865, "new_retained_MiB": 1.0141582489013672, "home_free_GiB": 50.00807571411133, "registered_live_jobs": ["V2_NATIVE_BANK_PLAN_FULL_TRAIN_IDLE"]}

失败处理：若全部授权TRAIN与既有日志仍无足够KEEP/APPLY监督，按任务书停止本轮结构探索，保留GMT+MLP。不会因还有GPU预算而增加Question、制造标签或改用封存数据。

最终路线：**P0_NATIVE_CLOSURE_RUNNING**。原MLP继续作为最强已测开发参考；科学目标未完成。
