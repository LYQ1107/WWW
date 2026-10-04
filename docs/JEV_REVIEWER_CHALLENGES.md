# JEV 审稿人质疑与预先回答

## “这只是把 threshold 换成 MLP”

回答必须展示：runtime legal action set、typed MATCH/MEMORY/REACTIVATION question、共享 option scorer、candidate permutation test、一次 reassociation contract 和长期 counterfactual target。只展示一个 accept/reject MLP 或 state-conditioned scalar threshold 不足以称为 JEV。

## “性能来自更多 GMT 调用”

所有方法报告 detector/GMT forward 次数、association rerun 次数、latency 和 peak GPU memory。`REASSOCIATE` 最多一次且只重跑既有 association/assignment，不重跑 detector。若 JEV 使用更多 compute，必须有 matched compute baseline。

## “oracle/future GT 泄漏”

ORACLE 只存在离线 counterfactual labeler。strict online process 禁止 import GT/evaluator，日志明确记录 `future_gt_access=false`；对未来 frame 截断、进程隔离和 source audit 都要有测试。

## “候选 ID 作为分类类别”

JEV 输出动作，不输出 proposal/track ID。track ID 只在 GMT commit adapter 中由受约束 assignment 生成。动作集合随 state 动态变化，masked softmax 只覆盖当前合法 action。

## “动作标签只是当前帧 correctness”

标签来自 H-step rollout utility，至少包含 AssA/IDF1、IDSW、Frag、contamination 和 recovery；报告 H=1 仅作诊断。不同 action 的 branch 从同一 state snapshot 开始，不能顺序污染 memory。

## “收益只是 threshold tuning”

必须包含固定阈值、全局 scalar、state-conditioned threshold、logistic gate、独立 MLP、shared encoder+heads 和 formal JEV。所有方法匹配 state fields、参数量、训练 split、Horizon 和 calibration budget，并做 same-score-different-state 分析。

## “概率不可解释/不校准”

报告 NLL、Brier、ECE、risk-coverage、abstention coverage，并在独立 calibration split 拟合 temperature。raw softmax concentration 不得直接命名为 confidence。

## “数据/评测不可复现”

每个结果绑定 dataset archive size/CRC、annotation JSON SHA256、GMT checkpoint SHA256、源 commit、config、seed、Horizon、utility weights 和 legal-mask schema。VisionTrack 原始文件、适配字段和缺失 GT 序列分别记录，不能静默覆盖。

## “IDF1 提升但 contamination 变长”

除了 HOTA/DetA/AssA/IDF1/MOTA/IDSW/Frag/CVIDF1/CVMA，必须报告 wrong commits、secondary switches、ID contamination duration、stale reactivation、recovery latency 和 new-ID count。任何长期污染恶化都要单独讨论。

## “一次数据集/一次随机种子偶然有效”

至少报告 sequence-held-out split、多个 seed、Horizon ablation、action permutation、legal-mask ablation 和 no-GT online replay。无法稳定复现时结论为 NO-GO，而不是挑选最佳 seed。

## 最终 GO/NO-GO

只有 oracle headroom、formal JEV 超过强 learned threshold/MLP、长期污染下降、校准与风险覆盖可接受、严格 online 无泄漏且结果稳定时才 GO。否则应保留审计和反例，明确结论为 NO-GO。
