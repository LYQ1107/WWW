# 同状态决策监督审计

阶段：BLOCKED_INSUFFICIENT_SUPERVISION。完整TRAIN12/13/14全部审计：True。
| Q2动作（非时间留出） | 认证完整计划 | 视频内人物连接依赖组 | video-person组 | 视频 |
|---|---:|---:|---:|---|
| APPLY_MLP | 8 | 7 | 8 | [12, 13, 14] |
| KEEP_GMT | 0 | 0 | 0 | [] |

| Q1类别 | 过去候选身份遭遇次数（有依赖） | 不同video/nativeID组 |
|---|---:|---:|
| RELIABLE | 37054 | 45 |
| CONFLICTED | 11382 | 19 |
| UNKNOWN | 54508 | 367 |

UNKNOWN占全部计划 1.1746%，占MATCH分歧配对样本 91.7526%。各行双方对错、合法候选覆盖/缺失与别名计数见 PAIRED_PLAN_OPPORTUNITY.json。
认证单位为完整计划。MATCH层先把改变的拒绝行保留UNKNOWN；新增原生闭包实际重放Bank/出生/WRITE，只有两条真实执行结果及严格过去唯一目标均可认证才解除UNKNOWN。两层记录分别保留。人物连接组包含当前GT人物与两方案候选owner，不把持续多帧事件当独立正例。
原始时间留出桶不用于Q2训练计数；过去GT认证跨桶连续，不是严格标签隔离。样本与分组元数据合计≤32MiB，不保存新完整训练库。
已有Rule/ORACLE事件不能改成Frozen-MLP的APPLY标签；历史重审计见 HISTORICAL_EVENT_REAUDIT.json。
真实Q2标签梯度诊断：{'1': {'gradient_norm': 2.3227092168461123, 'parameter_updates': 0, 'scope': 'CPU actual certified native-plan label smoke; not a trained Typed verifier'}}。这不等于训练成功；即使单一类别有梯度，也不能冒充双类独立监督充足。
完整原生计划见 NATIVE_PAIRED_PLAN_OPPORTUNITY.json；原MATCH预审计及逐行四象限见 PAIRED_PLAN_OPPORTUNITY.json。

完整原生闭包覆盖7577个计划状态，实际重放97处MATCH分歧；认证APPLY 8、KEEP 0、UNKNOWN 89。97处局部分支不等于97个独立人物。严格过去唯一目标之外的新人物出生/别名未解决状态没有被伪造为KEEP；本结论限于这套冻结认证标准，不宣称其他标注条件下也不存在可学任务。

全部63774行中，唯一可认证active目标31663行（49.6488%）；别名mask29180行（45.7553%）。已可认证唯一历史目标条件下active覆盖99.5035%，不能外推UNKNOWN目标。定义与各类别互斥计数见 DECISION_COVERAGE_DIAGNOSTICS.json。
