# GMT 决策路径审计

## 审计对象

本报告针对 `main@a436ce3e1ec9517323b79c7ebe455e0cce856bab` 的推理路径，重点是 `GTRCNN.sliding_inference_GMT`、`get_asso`、`run_global_tracker_plus`、`memory_bank` 和 `run_memory_tracker`。结论来自源码执行路径，不代表已经完成训练或评测。

## 路径图

```text
forward()
  └─ sliding_inference_GMT()
       ├─ 每个 frame/view: inference()
       │    └─ ROI heads -> proposal / bbox / reid_features
       ├─ 首帧: 初始化 track_ids
       │    └─ run_first_tracker_plus()
       └─ 后续 view:
            get_asso()
              └─ _forward_transformer()
            run_global_tracker_plus()
              ├─ _activate_asso()
              ├─ proposal -> historical track aggregation
              ├─ Hungarian assignment
              ├─ overlap_thresh accept/new-ID decision
              ├─ optional memory_bank()
              │    └─ get_asso() -> run_memory_tracker()
              └─ mutate track_ids / id_count / reid memory
```

`sliding_inference_GMT` 在 `gtr/modeling/meta_arch/gtr_rcnn.py:171-308` 中以 `view_frames = len(batched_inputs)/view_num` 展开多摄像头顺序；第 0 帧选择检测数最多的 view 作为初始轨迹来源（约 200-238 行），后续 view 使用 `run_global_tracker_plus`（约 257-280 行）。

## Evidence 产生

`gtr/modeling/roi_heads/gtr_roi_heads.py:199-244`：

1. 依据 `objectness_logits > asso_thresh` 筛 proposal；
2. 用 association ROI pooler 和 `asso_head` 产生 ReID embedding；
3. 推理时把 embedding 挂到每个 `Instances`；
4. 训练时可使用 `_get_asso_gt_reid` 和 appearance loss。

`gtr/modeling/roi_heads/gtr_roi_heads.py:482-528`：

1. 从 proposal 提取 box/time；
2. 可选位置/时间/相机 embedding；
3. transformer 接收拼接的 ReID feature；
4. 每个 decoder output 通过 `asso_predictor` 产生 association logits。

`gtr/modeling/roi_heads/gtr_roi_heads.py:541-549`：每一组 logits 追加一列 zero unmatched logit，沿列 softmax，然后丢弃 unmatched 列。因此 activated association 的每行总和小于等于 1；JEV 若使用该量必须同时知道被丢弃的 unmatched probability，不能把剩余列误认为完整分布。

## 主决策路径

`run_global_tracker_plus` 位于 `gtr/modeling/meta_arch/gtr_rcnn.py:420-500`。

### 1. proposal 重新分块

```python
asso_output = asso_output[-1].split(n_t[:-1], dim=1)
asso_output = self.roi_heads._activate_asso(asso_output)
asso_nonk = torch.cat(asso_output, dim=1)
```

当前待匹配窗口的最后一个实例集合是 query；其余集合是历史/上下文。query-to-context probability 被拼为 `n_k x Np`。

### 2. 轨迹聚合

历史 proposal 的 `track_ids` 被压成 `ids`。令 `id_inds[p,m] = 1` 表示 proposal p 属于轨迹 m，则：

```python
traj_score = torch.mm(asso_nonk, id_inds)
```

这把 proposal evidence 转成 query-to-track evidence。它不是 calibrated probability over tracks：同一历史 track 的长度会改变聚合量，且 unmatched probability 已被省略。

### 3. Hungarian 与固定阈值

对 `-traj_score` 做 Hungarian。对每个候选 `(query_i, track_j)`：

```python
thresh = overlap_thresh * track_length
```

除非 `not_mult_thresh=True`，否则 track length 会把阈值线性放大。只有 `traj_score > thresh` 才接受已有 track；否则 `track_ids[i]` 保持负值，之后创建新 ID。

因此当前决策同时依赖：association 总量、历史 track 长度、全局 threshold、Hungarian 的一对一约束。只调一个 scalar threshold 不能表达“同分不同状态”的决策差异。

### 4. memory 分支

`with_bank=True` 时，尚未分配 ID 的当前检测进入 `memory_bank`（约 460-479 行）。`memory_bank`（502-542 行）会：

1. 从 `poss_ids` 找不在当前 `unique_ids` 中的旧轨迹；
2. 用最近 `bank_size` 个 ReID feature 求平均；
3. 将旧轨迹摘要与当前 unmatched detection 合并；
4. 再次调用 `get_asso`；
5. 用 `run_memory_tracker` 和 `thred_bank` 再做一次 Hungarian/threshold。

这实际上已经有两个不同 threshold decision：主路径的 `overlap_thresh` 和 memory 路径的 `thred_bank`。JEV 实验必须把二者拆成两个 typed question，否则会把 memory write、reactivation 和 current association 混成一个标签。

### 5. commit 与状态突变

若没有接受或复活：递增 `id_count`、写 `id_count_dict[id] = 1`、保存 `id_reid_dict[id]`。若接受已有轨迹：更新 count、可能把 ID 加入 `poss_ids`，并将当前实例与历史 `Instances` 拼接到 `id_reid_dict`。

最后写入 `instances[k].track_ids`，并断言当前集合 ID 唯一。这里是唯一适合放 `commit_action` 的边界；JEV 头不应直接修改这些容器。

## 观测到的隐含状态风险

### 全局/模块状态

`sliding_inference_GMT` 每次开始时重置 `poss_ids.poss_ids`、`old_ids.old_ids` 和 `old_reids.old_reids`。这说明 tracker 不是无状态函数；离线 counterfactual branch 必须复制这些状态，不能在同一实例上顺序复用而污染下一个分支。

### 候选顺序

association 输出的列顺序来自 `instances` 拼接顺序，track aggregation 又依赖 `ids` 的顺序。任何 JEV legal action encoder 都必须附带稳定的 proposal/track key，并在 candidate permutation 测试中验证 action probability 不依赖列表位置。

### unmatched mass

`_activate_asso` 的 unmatched 列被删除，当前 `traj_score` 只用剩余 mass。日志和反事实状态应保留 unmatched probability，否则 JEV 可能把“没有足够 evidence”误判为低概率已有轨迹。

### memory mutation

`memory_bank` 会 `instances_old.reverse()`，会改变传入 list 的顺序；也会从 `poss_ids` 移除 ID，并修改 `old_reids`。在 shadow/oracle/counterfactual 模式中必须使用深拷贝或显式 immutable snapshot。

## 当前配置含义

`gtr/config.py` 中：

- `MODEL.ASSO_HEAD.ASSO_THRESH_TEST = -1.0` 控制测试 proposal 筛选入口；
- `MODEL.ASSO_HEAD.BANK_SIZE = 20` 控制 memory 摘要长度；
- `MODEL.ASSO_HEAD.THRED = 0.1` 是 memory reactivation threshold；
- `MODEL.ASSO_HEAD.WITH_BANK = False` 默认关闭 memory branch；
- `VIDEO_TEST.OVERLAP_THRESH = 0.1` 是主路径 threshold；
- `VIDEO_TEST.NOT_MULT_THRESH = False` 默认按 track length 放大 threshold；
- `REID = False` 是训练/association feature 配置开关，不等价于推理时是否存在 ReID evidence。

这些字段必须在 baseline manifest 中锁定。JEV/threshold/MLP 比较中不能让某个方法隐式改变 detector、association 或 memory 配置。

## 与 JEV 的最小对应关系

| 现有 GMT 操作 | JEV 研究中的角色 | 允许 JEV 做什么 |
|---|---|---|
| association logits/probability | observation/evidence | 作为 state feature；不能直接当 action label |
| Hungarian candidate | proposal | 作为第一候选；可请求一次 constrained reassociation |
| `traj_score > overlap_thresh` | 旧 gate | 作为 OFF/threshold baseline 或 state feature |
| unmatched current | question context | 选择 `START_NEW`、`REASSOCIATE`，必要时进入 memory question |
| memory bank query | reactivation proposal | 选择 `REACTIVATE_OLD` 或 `START_NEW` |
| `id_count_dict`, `id_reid_dict`, `poss_ids` | mutable tracker state | 只由 commit 层更新 |

目标接口应形如：

```text
state_t = snapshot(observation_t, gmt_evidence_t,
                   tracks_t, memory_t, disagreement_t)
question_t = typed_question(kind, state_t)
legal_actions_t = runtime_mask(kind, state_t)
action_dist_t = JEV(state_t, question_t, legal_actions_t)
action_t = select(action_dist_t)       # no candidate ID output
state_{t+1} = commit_gmt(state_t, proposal_t, action_t)
```

## 审计结论

现有 GMT 的核心 association/re-identification evidence 可以复用，但决策层不是可直接替换的一行 threshold。正式实现前必须完成：

1. OFF 模式的逐帧/逐 ID golden trace；
2. proposal、assignment、threshold、memory 的日志 schema；
3. legal action 和 typed question contract；
4. counterfactual state snapshot/restore；
5. Oracle gate 与强 learned-threshold/MLP 对照；
6. 只有在上述测试通过后，才实现 JEV head 和 commit adapter。
