# GMT-JEV 中文研究报告（本轮完整证据）

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


同约定原生最终输出：GMT HOTA68.389411、IDSW542、Frag586；Threshold HOTA68.541303、IDSW629、Frag582；Rule HOTA68.966066（+0.576655）、AssA69.235313、CVIDF188.912758、IDSW411（−24.17%）、Frag608（+3.75%）；Confidence HOTA68.642726、IDSW637、Frag568。Rule 次要指标出现正向，Threshold/Confidence 次要 IDSW 失败。stock 与 raw 均在结果前预注册，不能按结果换主要协议；stock 长期身份诊断已补齐，效率仍待验证。


原生 stock 持续量实测：回顾性 GMT4390、Rule3457（0.7875倍）；历史锚 GMT4046、Rule3768（0.9313倍）。主要 raw 结果仍保留，且全视频长度过滤使最终输出不构成在线因果证明。详细未知覆盖及删失语义见 STOCK_IDENTITY_TEMPORAL_AUDIT_ZH.md。可选归一化缓存保持当前所有控制器默认关闭，待全视频逐项等价审计后另测相同 Rule 的开销，不改变分数公式、权重或求解器。


新增结果后历史锚审计：仅使用更早场景已有至少3次已知观测、覆盖率>=80%的首个人物锚；屏蔽重复GT人物，未知/间断为右删失。认证错误人物关联持续相机帧confidence 3346、original 4054、rule 3792、threshold 4323。Threshold 相比 GMT 增加6.64%，Confidence 减少17.46%；各方法最长区间均778帧。不能用未来整段多数身份映射重写历史归属，也不把此描述量称作干预因果传播时间。参见 PAST_ANCHOR_TEMPORAL_AUDIT_ZH.md；这是结果后补查，未改写预注册门槛。

## 6. 普通方法是否足以解释收益

目前所有正向结果来自未训练的普通控制器；学习式架构独立优势未建立。MLP/Attention/CIC 没有通过冻结 TRAIN 信号前置条件，不进行盲目训练或架构消融。不能把零残差原型的效率测量冒充训练后的公平质量比较。严格认证 N01/N10 为0/1、0/2、0/1，覆盖有限、Bank结果部分 UNKNOWN，不是完整闭环归因。

## 7. 真实效率

参见 EFFICIENCY.md 与 SPEED_METRICS.json。只报告同一 GPU7 上三次真实完整视频的已完成结果；未完成项 PENDING。计时包含原生最终后处理，场景/图像FPS单位另列。GPU4 的前缀矩阵操作审计单独记录，不用于 FPS。全部三次计时已完成；各方法95%门槛以 SPEED_METRICS.json 的 efficiency_gate_pass 为准，质量与独立验证仍未通过。

## 8. WWW 科研资格

当前不能作为性能突破或 JEV 创新证明。缺联合质量优势、合格学习式对照、独立完整视频和种子复现。共享身份证据与跨相机条件化是待验证的机制假设，不能以命名替代对照。见 REVIEWER_ANALYSIS.md。

## 9. Goal 与下一步

应用 Goal 与仓库 MASTER_GOAL 均 ACTIVE，scientific_success=false。Rule 稳定方法 NO_GO，整个科学目标 NOT_ESTABLISHED。先完成已冻结的效率队列与 stock GMT 对照，不因开发正号追加阈值搜索；任何后续训练必须先有合格 TRAIN 信号。独立验证等待获准的未暴露范围，禁止新 DEV、sealed20–22、TEST、Full24。

## 10. 代码、证据与 GitHub

研究分支 research/gmt-jev-risk-association-20261014；[当前中文报告](https://github.com/LYQ1107/WWW/blob/research/gmt-jev-risk-association-20261014/docs/GMT_JEV/FINAL_RESEARCH_REPORT_ZH.md)。每阶段仅推源码、配置和紧凑JSON，权重、视频、图像、预测压缩包留在服务器并记录 SHA。实际发布终态由 runtime/publication/LAST_VERIFIED.json 的 local/remote HEAD 一致证明；报告生成来源 HEAD 与运行时源码 SHA 保存在 JSON。

当前资源快照：{"new_retained_MiB": 286.68444633483887, "home_free_GiB": 50.07233810424805, "GPU_wall_hours_including_live": 8.159367032243559}。只新增 home 文件，data1 读取历史资源；旧 NO_GO 与 checkpoint 未改写。

完整 TRAIN12/13/14 覆盖补查：31663 条可认证唯一合法候选中，GMT 错误 127 条。TRAIN 时间桶冻结 Rule 局部 N01/N10=1/0；原前缀报告保持不变，这一补查不代表原生闭环或独立验证。当前训练状态：NOT_RUN_NATIVE_TRAIN_RULE_SIGNAL_GATE_FAILED。见 FULL_TRAIN_OPPORTUNITY_AUDIT.json 与已冻结 TRAIN_NATIVE_RULE_VALIDATION_PROTOCOL.md。

完整原生 Rule 提交验证的 TRAIN 时间桶计数：{"N10": 1, "N10_new_identity_fragment": 1, "UNKNOWN_reference_Bank_or_birth": 19, "UNKNOWN_reference_owner": 1, "UNKNOWN_target": 24140, "actual_unique_target_commits": 27339, "certified_accepted_reference_rows": 27407, "changed_accepted_reference_commits": 51, "correct_reference_preserved": 27338, "eligible_unique_rows": 88, "reference_correct": 27339, "reference_wrong_person": 68, "rows": 51567, "unique_active_rows": 27427}。仅与同一当前真实状态的 accepted GMT MATCH 比较；原始拒绝后的 Bank 结果为 UNKNOWN，不作整段因果收益或 HOTA 声称。

## 完整 TRAIN 轨迹对照补查

输入图像、GT、有效配置、raw几何/置信度 SHA 在每段 Original/Rule 间一致。主要 raw 结果：

| 视频 | GMT HOTA | Rule HOTA | GMT IDSW | Rule IDSW |
|---|---:|---:|---:|---:|
| 12 | 76.396158 | 76.385165 | 123 | 145 |
| 13 | 85.769614 | 85.762773 | 56 | 61 |
| 14 | 79.847827 | 79.839220 | 298 | 327 |

合并官方评测主要差值：{"AssA": -0.013730191752586052, "CVIDF1": -0.025991896643986934, "DetA": -0.004039509721778245, "Frag": -2.0, "HOTA": -0.008858543771893324, "IDF1": -0.027643824676729878, "IDSW": 56.0, "MOTA": -0.07462129691815278}。三段 Rule HOTA 均下降，不选择有利序列，不用 TRAIN 替代独立验证；固定 Rule 没有稳定收益，原 TRAIN16 主协议失败仍成立。

受限 ORACLE 诊断合计：{"ALL_LEGAL_DIAGNOSTIC_ORACLE": {"HELDOUT_TEMPORAL_BLOCKS": {"E1_original_committed_wrong": 30, "E1_target_cannot_pass_active_MATCH_threshold_at_max_FP32_score": 4, "E1_target_edge_allowed": 30, "E1_with_any_allowed_edge": 30, "ORACLE_target_accept_on_E1": 3, "UNKNOWN_reference_Bank_or_birth": 4, "unique_active_rows": 5366}, "TRAIN_BLOCKS": {"E1_original_committed_wrong": 97, "E1_target_cannot_pass_active_MATCH_threshold_at_max_FP32_score": 9, "E1_target_edge_allowed": 97, "E1_with_any_allowed_edge": 97, "ORACLE_N01_accepted_reference": 3, "ORACLE_target_accept_on_E1": 17, "UNKNOWN_reference_Bank_or_birth": 19, "UNKNOWN_reference_owner": 3, "unique_active_rows": 26297}}, "CURRENT_GATE_ORACLE": {"HELDOUT_TEMPORAL_BLOCKS": {"E1_original_committed_wrong": 30, "E1_target_cannot_pass_active_MATCH_threshold_at_max_FP32_score": 8, "E1_target_edge_allowed": 1, "E1_with_any_allowed_edge": 5, "ORACLE_target_accept_on_E1": 1, "UNKNOWN_reference_Bank_or_birth": 4, "unique_active_rows": 5366}, "TRAIN_BLOCKS": {"E1_original_committed_wrong": 97, "E1_target_cannot_pass_active_MATCH_threshold_at_max_FP32_score": 31, "E1_target_edge_allowed": 11, "E1_with_any_allowed_edge": 25, "ORACLE_target_accept_on_E1": 7, "UNKNOWN_reference_Bank_or_birth": 19, "UNKNOWN_reference_owner": 1, "unique_active_rows": 26297}}}。所有 ORACLE 动作均未执行；这是有界合法动作空间见证，不是最优上限、神经网络可表达证明或在线收益。

单次后续修订已在 LEARNED_GATE_SMOKE_PROTOCOL.md 冻结：三个约5k参数小头使用同一 learned risk mode，保留原手工量为输入、保留±0.05残差及原生候选/求解器；先核验原生零残差与≤32MiB临时监督缓冲的契约，再决定是否执行一次同预算300步smoke。当前仍为0真实训练更新，不改变旧 Rule、旧模式或任何成功门槛。
