# Threshold 与 JEV 压力测试协议

## 目的

证明收益来自 typed state-transition decision，而不是更多参数、更多候选特征或一次偶然的阈值调节。所有方法使用冻结的 GMT evidence 和同一 candidate proposal。

## 对照组

### A. 原始固定阈值

原始 `OVERLAP_THRESH` 和 `THRED`，包括 `NOT_MULT_THRESH` 与 `WITH_BANK` 的固定配置。报告完整 baseline manifest。

### B. 全局 learned scalar threshold

训练一个标量 `tau`，在验证集上选择或学习；所有 state 使用相同 tau。禁止额外 state features。

### C. state-conditioned threshold

用同一个冻结 state encoder 输出标量 `tau(s)`，但最终动作仍是 `score > tau(s)`。参数量、隐藏维度、训练样本和 optimizer 与后续 MLP/JEV 对齐。

### D. logistic gate

对旧 threshold margin 或 association score 训练二分类 logistic gate；输出 accept/reject，不输出 typed action set，也不允许 reassociation。

### E. 独立 MLP policies

MATCH、MEMORY、REACTIVATION 各自独立 MLP；使用与 JEV 相同的 state feature budget、hidden size、训练 target 和 calibration split。

### F. shared backbone + separate heads

共享 state encoder，question type 通过 separate head；这可以区分“共享表示”与“真正 shared action scorer + runtime option set”。

### G. formal JEV

question/action conditioned shared scorer，runtime legal mask，显式 typed action，最多一次 reassociation，长期 counterfactual utility target。

## 必须匹配的资源预算

每个方法记录：

- trainable/frozen parameter count；
- forward FLOPs 或测得 GPU time；
- CPU/GPU memory peak；
- decision latency per current proposal and per frame/view；
- number of extra GMT association calls；
- number of memory reads/writes；
- training wall time and seed count。

Threshold 与 gate baseline 不得通过额外 detector/GMT rerun 获得信息；JEV 的 `REASSOCIATE` 也只能调用一次受约束的既有 assignment。

## 状态压力轴

在同一 checkpoint 上分层报告：

1. 高/低 proposal count；
2. high/low association entropy；
3. same raw score、不同 track age；
4. same raw score、不同 unmatched mass；
5. cross-view disagreement；
6. fresh track、stable track、stale memory；
7. single-candidate、empty-legal-action 和 masked-action cases；
8. detector/GMT evidence 缺失或 NaN recovery。

关键分析是“same-score-different-state”：固定旧 score 或 threshold margin，按 state bucket 比较 oracle action、错误提交、contamination duration 和 recovery。若 JEV 与所有 threshold 方法相同，不能声称使用了 state-transition information。

## 指标

主 tracking 指标：HOTA、DetA、AssA、IDF1、MOTA、IDSW、Frag、CVIDF1、CVMA。决策指标：

- action accuracy（相对 oracle label）；
- NLL、Brier、ECE、classwise calibration；
- risk-coverage/abstention；
- wrong action commits；
- contamination duration；
- secondary ID switches；
- stale reactivation；
- recovery latency；
- extra runtime/memory cost。

所有方法使用同一 strict online test。oracle action accuracy 只用于 offline validation，不计入最终 online tracking score。

## 通过/不通过条件

只有同时满足以下条件才允许继续主张 JEV：

1. ORACLE 在合法动作约束下有稳定 headroom；
2. formal JEV 超过强 state-conditioned threshold 和 matched MLP，而不依赖额外 GMT compute；
3. long-horizon contamination/IDSW/Frag 有下降，且 recovery 不以新 ID 爆炸为代价；
4. permutation、seed、Horizon 和 action imbalance 下结论稳定；
5. offline calibration 改善没有牺牲严格 online 性能。

任何条件失败都记录为 NO-GO，而不是通过改变 utility 权重或阈值范围事后修正结论。
