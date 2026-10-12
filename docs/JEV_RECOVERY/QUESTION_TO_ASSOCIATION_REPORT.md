# Question 到实际身份关联：补充诊断

状态：DEVELOPMENT_NOT_ESTABLISHED。不修改原 Gate、不重写 R2 标签、不改动正在运行的 R2/R3 模型。

当前训练先载入 R2_QUESTION_SUPERVISION 的 Cross 与旧 Continuity，然后在 jev_recovery_learning.load_records 中用 first-past-certified anchor 的 known_options/positive **替换 Stability**。因此早期 Cross/历史可靠性分歧与后期 Cross/Persistent 分歧不是同一个统计口径。旧审计和标签原文件均保留。

Cross 判断其他摄像头已经认证的纯历史是否支持当前人；Persistent 判断当前人是否符合此原生 ID 第一次过去认证的归属。混合历史在旧可靠性审计中为 0；当前 Persistent 已绑定的过去证书不会因此删除。未绑定 ID、未知当前人仍为 UNKNOWN。这个约定不认证真实出生 owner，也不证明未来安全。

当前两个问题分歧在 TRAIN12/13/14 分别为 47/349/97 条边；TRAIN16 为 4311。训练的 3601 个局部冲突组按共享 person/原生 ID 连边后只有 3 个保守依赖分量，不能称为 3601 个独立纠错事件。

辅助输出并未直接进入最终评分：edge_value 消费融合后的隐藏表示；plan_value 消费选中方案 token 的表示。故正例分类准确率与最终方案排名是不同目标。原 comparisons 排名还包含 0.25 Persistent 偏好；新增诊断另算仅 WHO 的排名，避免混淆。

已注册 R2_XIV_question_to_association_pilot_frozen_v1，使用同一 Pilot checkpoint、全部 2067 个真实 TRAIN16 状态、同一合法方案：完整、仅 Cross 表示、仅 Persistent 表示、简单平均、删除联合方案残差、置零辅助输出。前四项只作用于最后融合，不重新训练；辅助 Head 预测保持原值。额外置零 Head 输出要求最终分数逐位一致，以验证直接使用路径。

输出完整 N01/N10/UNKNOWN、Global ID 改配、两问题正确性与最终关联交叉表、WHO-only 与现有排名。最多保存 40 个两问题都正确而关联错误的事件，并列出合法正确替代、历史可用性、早期外观余弦、原始/修正分数。没有正确替代的事件不定义为排名失败。此同状态诊断不写原生状态；完整在线和提交证据须由独立在线实验补齐，不能伪装成已经完成。

显式辅助预测融合器与独立最终 Reader 若进行新训练，使用独立实验 ID 与预冻结配置，以及相同数据/预算的普通 MLP、Attention；冻结权重干预与重新训练结果分别报告。它们不得生成第四个大型版本。R3 强 XIII Fixed 原始和四种增量仍由既有队列执行。


## 已冻结的最小新增训练

R2_FUSION_*_v1 使用同一个已经训练完成、完全冻结的 R2 JEV Pilot 前端，比较显式辅助预测门控、两个独立最终 Reader、普通 MLP、普通 Attention。四种最终融合器参数为约 8.2 万，差距小于 1%；全部获得同样的两路表示、预测概率、原始余量和因果早期外观，使用同一 TRAIN12/13/14 批次、损失和预算。Tiny128 后 fresh Pilot1000，最后 checkpoint，全量 TRAIN16 冷启动原生状态评估。分类 Head 全部冻结，故新结果将直接检验相同辅助预测能否通过不同融合器转化为身份收益。

这是既有 R2 内的最终融合诊断，不是第四个大型版本，也不能称为从头训练的普通控制器对照。原来的纯 MLP/Attention 主对照仍由原队列完成。额外融合训练仅1000步，加共同 R2 前端1000步为2000步路径；不改变原正式4000预算。没有新增 DEV 调参。完整配置：DIAGNOSTIC_FUSION_PREREGISTRATION.json。
