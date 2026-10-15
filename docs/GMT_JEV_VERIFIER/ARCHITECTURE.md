# 架构与条件准入

完整GMT和5119参数冻结MLP保留；P0仅在相同原始轨迹的pre-commit分数/候选/历史包上离线比较，不在推理中使用GT。
Q1问题是过去身份历史是否可靠；Q2问题是保留完整GMT计划还是完整MLP计划。完整双问题架构只有监督合格后才允许训练、接入native提交。
当前Q2准入：False。899参数HistoryReliabilityHead仅实现24→32→3诊断；输入只保留因果历史通道，不实现Q2、不改变native commit。
不得把独立Q1分类头称为完整Typed JEV。Q1→Q2作用、共享问题状态、强制KEEP/APPLY和OFF全视频部署合同尚无新证据；P0标签合同不替代它们。
