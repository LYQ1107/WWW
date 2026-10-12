# 三个有界版本

R1 不改网络，扩展合法方案并按已认证冲突组采样。R2 保留每个具体 ID 的两类问题表示，跨相机 Reader 读取该 ID 的 other-camera 历史，连续性 Reader 读取 own-camera 历史，并复用同一 State 编码。F0 平均、F1 普通 MLP 融合、F2 条件 Reader 使用近似相同新增参数预算；另设纯 MLP、阈值与普通 Attention。F1 的前端包含同样的 typed Readers，因此只检验融合结构，不能单独当作纯普通关联器。R3 将经 TRAIN 机制认证的部分迁移冻结 XIII Fixed，所有强控制使用同样 Base。R2 设计在任何 R1 新 DEV 评价前冻结。
