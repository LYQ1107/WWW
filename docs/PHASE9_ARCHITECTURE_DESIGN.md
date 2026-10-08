# Phase IX architecture hypotheses — NOT IMPLEMENTED / NOT TRAINED

所有神经架构保持待检验设计，不能把生产 Torch producer 接口或开源代码审计称作 Candidate JEV 训练成功。G_NATIVE bridge FAIL，C–F 均 NOT_RUN。

共同输入是同一在线 state64、候选 evidence12、全合法 mask；references 只负责提交。候选共享编码器逐条输出价值，同时从 detection context 输出 semantic NEW。禁止绝对 ID embedding、候选 GT 保留、未来片段 attention。每个模型均调用同一 augmented Hungarian、同一 native commit，MEMORY/REACT 冻结。正确性估计与效用估计是不同头/监督，未执行 utility 保持 unknown。

| 假设模型 | 拟议区别 | 需要排除的混淆 |
|---|---|---|
| Candidate MLP | 相同逐候选共享 MLP + 公共竞争字段、独立 NEW head | 不弱化输入或让 JEV 独占上下文 |
| DeepSets | 同候选编码器、masked pooling、集合上下文广播再评分 | 与 MLP 同特征/监督/预算，排列等变 |
| CAMEL-inspired context | 前缀 token 和候选交互；轻量 masked context，保持在线约束 | 不照搬外部困难采样并将其当因果标签，不用未来 |
| Candidate JEV | Question/Action-conditioned candidate value、correctness/utility heads、竞争上下文 | 与无 Q/A、同上下文普通头消融；WHO 固定不能独立证明 question 有用 |

已冻结训练计划：seeds20261008/09/10，约34K参数参照，epochs5/20/50/100，独立组数据25/50/75/100%，raw/train-stat/LayerNorm，LR=.001、weight decay=.0001、batch32、gradient clip5。用户要求8K/34K/128K/500K容量曲线视最初 fair/tiny 结果有条件实施，不无条件笛卡尔积扩大训练。实际参数、MACs、forward latency均null，因为网络未实例化/训练/评测，不能把计划容量当测量值。

监督消融在同一网络上比较 correctness、实际 H32 utility、joint、frozen ranking/advantage；普通 MLP/DeepSets必须同样享有因果监督。先验证tiny可以拟合可靠标签、合法概率/NEW/finite gradients和训练线上 normalization一致，再做正式公平比较。NEW当前没有可靠正正确性标签，需要单独完成资格，不能伪造正例。全候选优先；TRAIN-only recall@8/16/32/64/full后再讨论Top-K，当前不截断也不虚构recall。

评价应分别报告 natural/hard-corrective/hard-negative/unknown，group-weight、视频配对和统计不确定性；使用强普通 baseline、同 solver 和同重求解预算，之后在sealed20/21/22真实 mutated online检验HOTA/AssA/IDF1/IDSW/MOTA、camera-frame传播时长、误纠正、birth/merge、效用regret和延迟。当前这些模型比较均未执行，不存在JEV架构优劣结论。只有MATCH闭环合格后，才研究真实WRITE如何影响身份查询以及relative OLD/NEW reactivation，Unified当前禁止启动。

**WHAT DID WE LEARN?** Candidate conditioning、上下文交互和共享求解器都是普通可学习关联也具备的能力；JEV独立贡献必须经Q/A消融与公平监督比较，不能由名称或结构示意替代。
