# JEV Decision Oracle Gate

## 目的

本文件定义 GMT 决策层的 OFF、SHADOW、ORACLE 和后续 JEV 模式，避免把 future ground truth 泄漏到严格 online test。ORACLE 不是最终模型，也不是训练时随意读取未来标签的 shortcut；它只是回答一个问题：在同一冻结 GMT evidence 和同一合法动作集合下，动作选择本身是否有足够的长期 headroom。

## 冻结条件

所有模式必须共享：

- 同一 detector/backbone；
- 同一 Stage1 REID checkpoint；
- 同一 GMT Stage2 checkpoint；
- 同一输入图片、proposal、association logits、窗口和 memory 配置；
- 同一 frame/view 顺序、随机种子和后处理；
- 同一合法动作生成器和 candidate proposal。

唯一允许变化的是 decision policy。若 action policy 改变了 detector、proposal 数量、association forward 或 memory capacity，结果不能归因于 JEV。

## 模式契约

### OFF

OFF 直接调用审计前的 GMT commit 逻辑：Hungarian、`OVERLAP_THRESH`、`WITH_BANK`、`THRED` 和新 ID 分支均保持原语义。它必须输出逐帧 golden trace：

```text
(frame, view, proposal_key, candidate_track,
 old_action, threshold_score, threshold_value,
 committed_track_id, memory_read/write)
```

OFF 是回归基线。若重构后的 OFF 与原 GMT 不逐帧一致，不能进行任何学习型比较。

### SHADOW

SHADOW 运行同一个 JEV/Oracle interface，但不改变 GMT 状态。它保存：

- typed question；
- legal action mask；
- action probabilities/logits；
- proposed action；
- OFF action；
- state digest 和运行耗时。

SHADOW 可以在真实 online stream 上运行，因为它不读取未来 GT，也不提交 JEV action。它用于检查 action coverage、calibration 和与旧 threshold 的 disagreement。

### ORACLE

ORACLE 只允许在 offline train/validation counterfactual builder 中使用 future GT。它接收与 JEV 完全相同的 state、proposal 和 legal action set，对每个合法动作独立分支执行 H-step rollout，然后用固定 utility 选择：

```text
a* = argmax_a U_H(rollout(state_t, a), future_gt[t:t+H])
```

如果多个动作 utility 相同，使用固定 tie-break（优先不污染现有轨迹，再优先短路径）；tie-break 必须记录。ORACLE 不输出 proposal/track ID，只输出三种 typed question 的 action label 或 soft utility target。

### JEV

JEV 只接收在线可得 state 和 legal mask。严格 online test 时不得读取 GT、未来 frame 的标注、序列最终长度或 evaluator feedback。JEV 可以输出 action distribution 和 abstention，但 commit 仍由 GMT adapter 完成。

## 三种 typed question

### MATCH

合法动作：

- `ACCEPT_CURRENT`
- `REASSOCIATE`（每个决策最多一次）
- `START_NEW`

`REASSOCIATE` 的分支屏蔽第一候选 proposal，使用已经计算的 detector/proposal evidence 重新执行 GMT association/assignment；禁止 detector rerun，禁止 JEV 直接指定第二 proposal。

### MEMORY

合法动作：

- `WRITE_MEMORY`
- `SKIP_MEMORY`

此问题只决定是否把当前 unmatched 状态写入 memory contract；memory key、容量和 feature 聚合仍由 GMT 维护。

### REACTIVATION

合法动作：

- `REACTIVATE_OLD`
- `START_NEW`

如果当前没有合法旧轨迹，`REACTIVATE_OLD` 必须被 mask 掉。旧 ID 由 GMT assignment 根据 JEV action 做提交，JEV 不负责生成 ID。

## Counterfactual rollout 规则

每个训练/验证 state 保存一个可恢复快照：

```text
tracker_state = {
    track_ids,
    id_count,
    id_count_dict,
    id_reid_dict,
    poss_ids,
    old_reids,
    memory_bank,
    current_frame/view cursor,
}
```

每个 candidate action 从同一快照开始，执行相同的未来 detector/GMT evidence。分支只允许改变当前 question 对应的 action；不能让一个候选分支先看到另一个分支的 memory write 或 ID assignment。

推荐先实现 H ∈ {1, 4, 8, 16} 的离线 rollout，并把 H 作为数据字段和报告维度。未来 GT 只存在于 data-builder/labeler 进程，最终 `.jsonl` 不应包含未脱敏的未来输入路径之外的 evaluator oracle 逻辑。

## Utility 与标签

动作效用不能只用当前帧的 correctness。最小记录：

- `AssA`, `IDF1`, `MOTA` 或 frame-local association score；
- `IDSW`, `Frag`；
- ID contamination duration；
- stale reactivation 和 recovery latency；
- memory write/skip 成本；
- optional runtime/extra association cost。

一个可审计的标量形式是：

```text
U_H = w_assa * ΔAssA
    + w_idf1 * ΔIDF1
    - w_idsw * ΔIDSW
    - w_frag * ΔFrag
    - w_contam * contamination_duration
    + w_recovery * recovery_gain
    - w_cost * extra_runtime
```

权重、H、tie-break、不可用 action 的处理必须写入 manifest。训练第一阶段可用 best-action CE；第二阶段使用归一化 utility posterior 的 KL/soft target；任何 outcome fine-tuning 都必须保持 validation state 隔离。

## Oracle headroom 判据

在正式训练 JEV 前，先计算：

1. `ORACLE - OFF` 的 HOTA/DetA/AssA/IDF1 和 IDSW/Frag 改善；
2. oracle action coverage（每个 question/action 的频率）；
3. 不同 H 下的 headroom 稳定性；
4. 同一 raw threshold score、不同 state 下的 oracle action 分歧；
5. 只允许一次 reassociation 的 oracle 上界。

若 ORACLE 在严格约束下没有稳定 headroom，JEV 没有研究价值，应该先修正 evidence/action contract，而不是训练更大的网络。

## 防泄漏检查

- online `forward`、`inference`、JEV head 和 commit adapter 不得 import evaluator/GT reader；
- oracle/counterfactual 代码必须放在离线工具命名空间，且 manifest 标出 `uses_future_gt=true`；
- strict test 的 process environment 中关闭 oracle 变量；
- 每条 decision log 记录 `mode` 和 `future_gt_access=false/true`；
- 对未来 frame 做截断测试，确认 online 输出不变化；
- 用同一 state 重放，确认 action probability 不依赖 rollout 顺序。

## 结论

Oracle gate 先于 JEV head。它只要证明在冻结 GMT 下动作层确实有长期 headroom，就足以支持继续实现；它不能成为最终评测模式，也不能让 JEV 通过隐藏 future GT 获得结果。
