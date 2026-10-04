# MLP 与 formal JEV 的公平压力测试

## 研究问题

强 MLP 可以从 state feature 学会一个复杂 gate。为了证明 formal JEV 的收益不是参数规模或 feature engineering，需要把 MLP 对照做完整，并让它在相同冻结 GMT evidence、训练 target、数据 split、Horizon 和计算预算下工作。

## 对照结构

1. **Scalar gate**：单个可学习阈值。
2. **Logistic gate**：旧 threshold margin 到 accept/reject。
3. **Independent MLP**：每种 question 一个 MLP，固定 action rows。
4. **Shared encoder + separate heads**：共享 state encoder，但每种 question/action head 独立。
5. **Shared MLP with padded action slots**：一个 MLP 对固定最大动作数输出，runtime mask 后 softmax。
6. **Formal JEV**：shared option/action scorer，typed question/action embedding，runtime legal set，permutation test 和 explicit abstention。

第 5 组是最重要的反例：它可以拥有类似的参数量，但 fixed slot identity 容易学习候选位置，不能自动视为 JEV。

## 参数匹配

报告每个实现的：

- state encoder hidden size/layers；
- option/action embedding size；
- total trainable params；
- optimizer、learning rate、weight decay、epoch/step；
- batch、sequence grouping、Horizon；
- temperature/calibration 参数；
- inference latency、peak memory、额外 GMT calls。

如果 JEV 因为 action set 变长而减少参数，需报告每个 state 的平均/最大 K 和 padded MLP 的实际 padding 成本。

## 输入公平性

每个对照只能使用同一 `state_digest` 对应的输入字段：

- observation；
- GMT association evidence；
- track/memory summary；
- old threshold score/action；
- disagreement。

不得给 MLP 额外的 future label、proposal ID、序列名或按位置编码，而 JEV 不给；也不得把 JEV 的 typed action semantics 隐藏在 MLP 的手工 feature 中。

## 关键诊断

### Same-score / different-state

按旧 threshold score 分箱，在每个箱内再按 track age、unmatched mass、memory age、view disagreement 分层。比较各方法：

- oracle action distribution；
- action entropy/calibration；
- wrong commit；
- contamination duration；
- recovery latency。

如果 state-conditioned MLP 能在这些分层上达到 formal JEV 的全部效果，报告 NO-GO 或至少说明 JEV 结构没有独立增益。

### Candidate permutation

对相同 state 随机置换合法 action 的输入顺序，要求输出概率按同一置换变换。固定-slot MLP 若不能通过，不能与 permutation-equivariant JEV 直接比较而不说明限制。

### Legal-mask ablation

逐个删除合法 action，确认所有方法都不会给被删除 action 非零质量。特别检查空集合、单动作集合和只有 `START_NEW` 的状态。

### Long-horizon corruption

构造第一步错误接受、错误 reactivation、错误 memory write 的状态，测量 H 帧后的 ID contamination、secondary ID switch 和 recovery。frame-local accuracy 高但 contamination 长的方法不能被判为优胜。

## 结论规则

正式 JEV 只有在 matched MLP、state-conditioned threshold、共享 encoder heads 都报告后，且在相同资源下稳定降低长时域风险，才可进入主结果。只报告 JEV 相对固定 threshold 的提升不够。
