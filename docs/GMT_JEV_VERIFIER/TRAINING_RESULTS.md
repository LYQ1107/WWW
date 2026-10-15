# 真实训练记录

Q1阶段：PENDING_Q1_OR_ELIGIBILITY。Typed/ordinary/Q2/GMT/原MLP新增更新均为0。

Q1若训练只使用TRAIN12/13非留出桶，固定300步；TRAIN14只作Q1开发诊断。当前GT不作为历史特征，UNKNOWN仅为过去证书不充分的三分类标签，不是动作伪标签。

完整逐25步loss/梯度、权重SHA、类别支持与混淆矩阵见 TRAINING_METRICS.json / Q1_RELIABILITY_DIAGNOSTIC.json。
Q1分类成绩不表示已改善HOTA、通过Q2机制或实现因果身份保护。
