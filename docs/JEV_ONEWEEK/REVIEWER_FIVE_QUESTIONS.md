# 五个审稿人问题：所有实验强制检查

每次计划登记须引用下表问题、同条件契约、结果和限制。Attention或多Head的存在不能替代独立价值证据。

| 问题 | 必需实验 | 当前状态 |
| --- | --- | --- |
| Q1：JEV 与二次关联有什么区别？ | 原 GMT、规则二次、重复Hungarian、学习二次、新JEV及评分×求解2×2；额外求解数、候选范围和历史分开归因 | NOT_RUN |
| Q2：为什么固定或可学习阈值不够？ | 固定、state阈值、pair阈值、Binary JEV、相对候选关系；同合法集与历史，检查决策能否被阈值重现 | NOT_RUN |
| Q3：为什么融合、MLP 或普通 Attention 不够？ | 固定/线性融合、MLP、DeepSets、Attention、Fixed、Multi；相同全部证据/监督/更新/求解，披露继承共同初始化 | NOT_RUN |
| Q4：增加多少参数和成本？ | 真实加载与forward-active计数、替换GTA净变化，按同GPU/真实双相机完整视频三次重启分解参数/FLOPs/显存/scene FPS/p50/p95 | NOT_RUN |
| Q5：是否符合 JEV 思想？ | State一次、两个可独立验证Question、动态Options、概率不确定性与共享计算；固定/单问题/无共享/无Reader/无cross/无stability/顺序/数量/敏感性/校准 | NOT_RUN |
