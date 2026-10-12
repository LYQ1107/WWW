# JEVTrack 一周科研执行目标

研究跨摄像头证据是否能够提高 Global ID 关联准确性并保持长期身份稳定。采用 Phase XIV 算法与 Phase XV 原生接口，先恢复基线和同条件比较，再验证跨摄像头一致性与可观察身份连续性。真实结果可以否定 JEV 假设。

本周独立 MATCH-only 协议允许已有 TRAIN12/13/14/16 受控关联研究，不构成原 MASTER 长期安全纠错监督门槛通过。新模块梯度限 TRAIN12/13/14，16 仅作已暴露控制器审计；DEV17/18/19 仍是反复使用的开发集。20/21/22、官方 TEST 和 Full24 保持封存。已训练初始化对16及前端对全部 TRAIN 的暴露必须披露。

128-step Tiny → 1000-step Pilot → 至多4000-step正式预算，三个固定种子20261008/09/10。只训练新增模块，所有普通强对照获得相同证据、监督、方案和更新预算。至多三轮主要算法修改，不因第一轮科学失败结束，也不无限扫参。新增留存上限8 GiB，两分区至少保留30 GiB。每个完成阶段推送必要代码、配置、失败、指标和 SHA；模型/图像/轨迹留服务器。

全部实际模型训练、完整 native mutated-state 在线结果、完整视频效率和五个审稿人问题证据都完成后，才形成科学结论。阶段完成不能冒充独立 JEV 贡献成立或原 MASTER 已完成。执行状态见 reports/JEV_ONEWEEK/EXPERIMENT_REGISTRY.json。

官方研究轨日期已核验：[WWW2027 Research Track](https://www2027.thewebconf.org/research-track-papers/)，摘要10月18日、全文10月25日（AoE）。
