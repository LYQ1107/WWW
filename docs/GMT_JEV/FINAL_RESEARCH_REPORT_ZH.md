# GMT-JEV 中文研究报告（当前证据，效率任务仍以 JSON 终态为准）

## 1. 完整 GMT 基线

TRAIN16 单种子、1034 场景帧、2068 张真实图像；原始完整 Stage2 参数 53,515,309。HOTA 68.054089、AssA 67.064493、CVIDF1 86.544619、IDF1 87.182280、IDSW 1898、Frag 503、MOTA 87.000434。该视频为已暴露开发视频，与历史 DEV17–19 的 78.110 不可直接比较。

原项目源码冻结在 04324354；模型严格加载完整 Stage2，checkpoint SHA256 cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8。VFCE 1024、RPCE 128、GTA 1152。Original 与 Adapter OFF 的32帧和完整视频 logits、分数、提交、预测一致；零残差 CIC 的32帧原生等价也通过。参见 GMT_BASELINE_AUDIT.md 和 BASELINE_PARITY.json。

## 2. 可纠正错误机会

TRAIN12/13/14 冻结前缀共14541行；11305唯一认证行中25行不是当前合法目标，其中23行有另一摄像头历史。25行来自14个时间块；大量别名/短历史仍是 UNKNOWN。规则和单次有界校准探针均未产生认证接受纠错，未满足训练前置条件。初版拒绝接口的11次伪变化已有独立审计记录，旧结果保留。

## 3. 最小 CIC 输入与网络

post-GTA 24个观测特征：分数竞争、历史支持、年龄、相机内/间余弦、冲突与检测置信度。共享摘要仅来自已提交的 native ID。CIC 使用32维 edge/query/context 和两个观测冲突角色，4962参数；MLP5119、Attention4898。有界归一化残差上限0.05，乘原生 support count 加入 summed score，再由同一原生匹配/阈值处理。GT仅用于离线标签和评测。预注册 CE、preserve 与 residual loss 已定义，但没有执行训练，新增梯度更新为0。

## 4–5. HOTA 与身份指标

Threshold HOTA +0.135169，IDSW1898、Frag499；Rule HOTA +0.502582、AssA +0.836247、CVIDF1 +0.973228，但 IDSW2191（+15.44%）、Frag503；Confidence HOTA +0.261053、AssA +0.495980、CVIDF1 +0.376788、IDSW1839、Frag500。完整数表见 ONLINE_RESULTS.md。

Rule 未通过稳定性门槛；Confidence 尚未达到 +0.5。所有原始检测记录去除 track_id 后逐项一致。身份不一致持续量（回顾性相机帧）GMT5274、Threshold5090、Rule4680、Confidence5071；不能用持续量下降抵消 Rule 的 IDSW 与额外身份恶化。


新增结果后历史锚审计：仅使用更早场景已有至少3次已知观测、覆盖率>=80%的首个人物锚；屏蔽重复GT人物，未知/间断为右删失。认证错误人物关联持续相机帧confidence 3346、original 4054、rule 3792、threshold 4323。Threshold 相比 GMT 增加6.64%，Confidence 减少17.46%；各方法最长区间均778帧。不能用未来整段多数身份映射重写历史归属，也不把此描述量称作干预因果传播时间。参见 PAST_ANCHOR_TEMPORAL_AUDIT_ZH.md；这是结果后补查，未改写预注册门槛。

## 6. 普通方法是否足以解释收益

目前所有正向结果来自未训练的普通控制器；学习式架构独立优势未建立。MLP/Attention/CIC 没有通过冻结 TRAIN 信号前置条件，不进行盲目训练或架构消融。不能把零残差原型的效率测量冒充训练后的公平质量比较。严格认证 N01/N10 为0/1、0/2、0/1，覆盖有限、Bank结果部分 UNKNOWN，不是完整闭环归因。

## 7. 真实效率

参见 EFFICIENCY.md 与 SPEED_METRICS.json。只报告同一 GPU7 上三次真实完整视频的已完成结果；未完成项 PENDING。计时包含原生最终后处理，场景/图像FPS单位另列。GPU4 的前缀矩阵操作审计单独记录，不用于 FPS。当前未测完，不能声明95%效率门槛通过。

## 8. WWW 科研资格

当前不能作为性能突破或 JEV 创新证明。缺联合质量优势、合格学习式对照、独立完整视频和种子复现。共享身份证据与跨相机条件化是待验证的机制假设，不能以命名替代对照。见 REVIEWER_ANALYSIS.md。

## 9. Goal 与下一步

应用 Goal 与仓库 MASTER_GOAL 均 ACTIVE，scientific_success=false。Rule 稳定方法 NO_GO，整个科学目标 NOT_ESTABLISHED。先完成已冻结的效率队列与 stock GMT 对照，不因开发正号追加阈值搜索；任何后续训练必须先有合格 TRAIN 信号。独立验证等待获准的未暴露范围，禁止新 DEV、sealed20–22、TEST、Full24。

## 10. 代码、证据与 GitHub

研究分支 research/gmt-jev-risk-association-20261014；[当前中文报告](https://github.com/LYQ1107/WWW/blob/research/gmt-jev-risk-association-20261014/docs/GMT_JEV/FINAL_RESEARCH_REPORT_ZH.md)。每阶段仅推源码、配置和紧凑JSON，权重、视频、图像、预测压缩包留在服务器并记录 SHA。实际发布终态由 runtime/publication/LAST_VERIFIED.json 的 local/remote HEAD 一致证明；报告生成来源 HEAD 与运行时源码 SHA 保存在 JSON。

当前资源快照：{"new_retained_MiB": 255.44586849212646, "home_free_GiB": 50.166725158691406, "GPU_wall_hours_including_live": 2.0288683084522683}。只新增 home 文件，data1 读取历史资源；旧 NO_GO 与 checkpoint 未改写。
