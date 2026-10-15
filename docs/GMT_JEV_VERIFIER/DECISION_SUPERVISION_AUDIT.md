# 同状态决策监督审计

阶段：P0_RUNNING。完整TRAIN12/13/14全部审计：False。
| Q2动作（非时间留出） | 认证完整计划 | 人物连接依赖组 | video-person组 | 视频 |
|---|---:|---:|---:|---|


UNKNOWN占全部计划 0.0000%，占实际有分歧计划 0.0000%。各行双方对错、合法候选覆盖/缺失与别名计数见 PAIRED_PLAN_OPPORTUNITY.json。
认证单位为完整计划；任一改变行的Bank/出生反事实不明就保留UNKNOWN。人物连接组包含当前GT人物与两方案候选owner，不把持续多帧事件当独立正例。
原始时间留出桶不用于Q2训练计数；过去GT认证跨桶连续，不是严格标签隔离。样本与分组元数据合计≤32MiB，不保存新完整训练库。
已有Rule/ORACLE事件不能改成Frozen-MLP的APPLY标签；历史重审计见 HISTORICAL_EVENT_REAUDIT.json。
真实Q2标签梯度诊断：PENDING。这不等于训练成功，空字典表示没有可用认证动作标签。
