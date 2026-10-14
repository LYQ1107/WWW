# FINAL GOAL — GMT-JEV

GMT-JEV: Task-Adaptive Cross-Camera Identity Reasoning for Efficient Global Multi-Target Tracking

版本 GMT_JEV_20261014_V1，状态 ACTIVE。来源提交 `04324354e9e9971eec6d937b961590a0cdbf13a9`；授权来源与时间见 MASTER_GOAL.json。

原始完整 GMT 是唯一必须超越的主要外部基线。完整保留检测、VFCE/ReID、RPCE、GTA Transformer、原始关联分数、合法候选、全局匹配、Gallery/Bank、原始 REACT/WRITE。禁止 JEV_DIRECT 替代 GTA、GMT 重训、新数据下载、完整特征缓存重建或大规模数据集构建。

在 GTA 聚合当前 Global ID 分数之后，研究极小残差和跨摄像头证据冲突门控；Adapter OFF 和零残差必须严格等价，最终仍由原生求解器与身份管理提交。

成功门槛：匹配范围 HOTA 至少 +0.5，AssA/CVIDF1 不下降；IDSW、Frag、长期身份不一致不超过1.05倍；同卡三轮完整图像平均FPS不低于GMT的95%；对普通Threshold/MLP/Attention及同求解器二次关联有公平优势。论文级科学成功还需获准的独立完整视频与跨种子复现。

授权TRAIN12/13/14用于小规模MATCH研究；TRAIN16仅开发诊断，不能伪称独立验证。DEV17/18/19不新增选模/评估；20/21/22、TEST和Full24不解封。旧D0/D1不是GMT训练样本。

本轮预算12 GPU-wall-hours、全部新增磁盘留存≤512 MiB、目标盘余量≥30 GiB。data1当前低于门槛，不新增实验文件；本轮在home盘执行。新工作区约242 MiB，后续运行留存将受剩余总额约束。

旧路线 SUPERSEDED：旧 JEV_MASTER、Recovery、Fastcheck 的结果、checkpoint 和NO_GO都保留。用户已清除应用内旧 Goal，新的 GMT-JEV Goal 已注册为 active，与本文件及机器入口一致；旧研究结果未被改写为成功。
