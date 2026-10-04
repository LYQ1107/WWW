# GMT 到 JEV 的代码映射

## 审计基线

- 仓库：`LYQ1107/WWW`
- 分支：`main`
- 基线 commit：`a436ce3e1ec9517323b79c7ebe455e0cce856bab`
- 当前审计为只读；没有把 JEV 代码插入 GMT，也没有改变 checkpoint 或默认配置。

下表按实际执行路径描述现有 GMT 的责任边界。行号以基线工作树为准。

| 阶段 | 文件/行 | 现有行为 | JEV 接口候选 |
|---|---|---|---|
| 模型入口 | `gtr/modeling/meta_arch/gtr_rcnn.py:82-129` | 推理从 `forward` 进入 `sliding_inference_GMT`；训练走 backbone、proposal generator、ROI heads | JEV 只放在推理 decision boundary；训练期先保持 GMT checkpoint 冻结 |
| 多摄像头滑窗 | `gtr/modeling/meta_arch/gtr_rcnn.py:171-308` | 按 frame/view 组织实例，首帧初始化 ID，后续逐 view 调 `get_asso` 和 tracker | 每个待处理 view/frame 形成一个 typed decision state；状态中保留窗口历史和 memory 摘要 |
| proposal/ReID evidence | `gtr/modeling/roi_heads/gtr_roi_heads.py:199-244, 457-463` | objectness 筛 proposal，ROI pool 后产生 `reid_features`；推理把特征挂到 Instances | 这些是 observation/GMT evidence，不应被 JEV 重新学习为候选 ID |
| association transformer | `gtr/modeling/roi_heads/gtr_roi_heads.py:482-528` | 用 box/time position 和 transformer 产生 proposal-to-context logits；训练有 association GT loss | JEV 输入 association probability、边缘、top-2 gap、时空信息、历史一致性和 disagreement |
| probability activation | `gtr/modeling/roi_heads/gtr_roi_heads.py:541-549` | 追加 unmatched zero column 后 softmax，再移除 unmatched 列 | 保留原始 logits/probability 给 state；JEV legal action mask 不应复用 proposal 行作为动作类别 |
| association 调用 | `gtr/modeling/meta_arch/gtr_rcnn.py:354-366` | 拼接实例的 ReID feature，调用 `_forward_transformer`，返回 association output/boxes/计数 | `PROPOSE` 阶段；可在 REASSOCIATE 时复用同一 detector 输出、重新跑 association |
| 首帧跨 view | `gtr/modeling/meta_arch/gtr_rcnn.py:200-251, 310-352` | 首帧先选检测最多的 view 建初始 ID，随后 `run_first_tracker_plus` 赋值 | 作为 separate initialization protocol；不能用后续 JEV 结果反向改变首帧基线 |
| 当前 tracker 主决策 | `gtr/modeling/meta_arch/gtr_rcnn.py:420-500` | 对 proposal association 概率按已有 track ID 聚合成 `traj_score`，Hungarian 后以 `OVERLAP_THRESH` 接受，否则新 ID；可进入 memory bank | 这里是主 `PROPOSE → JEV DECIDE → COMMIT` 重构点 |
| memory write/reactivation | `gtr/modeling/meta_arch/gtr_rcnn.py:460-479, 502-542` | 未匹配当前检测进入 memory 查询；旧 ReID 平均后重新调用 association，再由 `thred_bank` 决定复活 | 单独的 MEMORY/REACTIVATION question；`WRITE_MEMORY`/`SKIP_MEMORY` 和 `REACTIVATE_OLD`/`START_NEW` 必须是动作 |
| memory assignment | `gtr/modeling/meta_arch/gtr_rcnn.py:545-577` | memory association 按 track ID 聚合，使用 `thred_bank` 接受，并更新全局状态 | JEV 不能直接输出 old ID；先输出 action，再由受约束的 GMT assignment 提交 ID |
| 配置 | `gtr/config.py:13-42, 83-95` | `ASSO_*`, `BANK_SIZE`, `THRED`, `WITH_BANK`, `REID`, `OVERLAP_THRESH` 控制现有流程 | 新增 shadow/off/oracle/JEV mode 和日志配置时，默认值必须保持 OFF 等价 |

## 现有状态与 evidence

对一个待处理的当前检测集合 `D_t`，GMT 已经能提供以下可复用信息：

1. 当前 proposal 的 bbox、objectness、ReID embedding、view/time；
2. 当前 proposal 到窗口内 proposal 的 association logits/probability；
3. 将 association 按历史 `track_ids` 聚合后的 `traj_score`；
4. Hungarian assignment 的候选配对、top-1/top-2 gap 和 assignment margin；
5. 当前 track 的长度、最近命中/丢失、历史 ReID 摘要和 memory age；
6. 现有阈值判定与 memory 查询之间的 disagreement；
7. detector/GMT 的运行耗时、候选数量、空候选和跨 view 一致性。

这些量构成 observation/evidence，不等同于 action label。特别是 `track_ids` 是 GMT 的内部状态，不是 JEV 的固定分类表。

## 现有融合点为何不是 JEV

`run_global_tracker_plus` 当前把四种不同语义揉在一个过程里：

1. `_activate_asso` 把 association logits 变成 proposal probability；
2. `traj_score = asso_nonk @ id_inds` 把 proposal probability 聚合成轨迹分数；
3. Hungarian assignment 产生一对一候选；
4. `traj_score > overlap_thresh * track_length` 决定接受还是新 ID；
5. `with_bank` 分支继续查旧 memory，并用另一个阈值决定复活；
6. 未接受的 detection 直接递增 `id_count` 并写入 `id_reid_dict`。

这段代码没有 typed question、运行时 legal action set、显式 abstention、长期 utility，也没有区分“提出候选”和“提交状态转移”。所以直接在阈值旁加一个 MLP 不足以实现 JEV。

## 目标分层

### Association question

状态包含当前 observation、GMT association evidence、历史轨迹摘要和 disagreement。首先生成合法动作集合：

- `ACCEPT_CURRENT`：接受当前第一候选；
- `REASSOCIATE`：拒绝第一候选，最多一次，屏蔽被拒 proposal 后重新跑 GMT assignment；
- `START_NEW`：不接受当前历史轨迹，为检测创建新轨迹。

`REASSOCIATE` 不能触发 detector rerun，也不能让 JEV 直接选择第二个 proposal；第二次 association 必须由原 GMT 过程在约束候选集上完成。

### Memory question

对于 unmatched current detection 和可写入的 track state，动作集合至少包含：

- `WRITE_MEMORY`；
- `SKIP_MEMORY`。

写入的对象、长度和容量由 GMT memory contract 决定；JEV 只决定动作，不直接修改 `id_reid_dict`。

### Reactivation question

对 memory retrieval 产生的候选，动作集合至少包含：

- `REACTIVATE_OLD`；
- `START_NEW`。

如果没有合法旧轨迹，`REACTIVATE_OLD` 必须从 mask 中消失，而不是保留一个永远失败的类别。

## 建议的最小重构边界

实现时应保持以下纯函数式边界，便于 OFF 对照和 counterfactual rollout：

```text
GMT evidence
    -> propose_association(state)
    -> build_legal_questions(state, proposal)
    -> JEV decide(question, legal_actions)
    -> commit_action(state, proposal, action)
    -> next_state / diagnostics
```

其中：

- `propose_association` 不看 JEV logits；
- `decide` 不产生 candidate ID，不执行 assignment，不写 memory；
- `commit_action` 是唯一允许修改 `track_ids`, `id_count_dict`, `id_reid_dict`, memory 的位置；
- OFF 模式必须调用原始 commit 路径，保证与旧 GMT 完全一致；
- SHADOW 模式运行 JEV 但不提交，记录 counterfactual action 和概率；
- ORACLE 模式使用 future GT 只生成离线 oracle gate，不进入严格 online test。

## 必须先建立的测试钩子

在修改 `run_global_tracker_plus` 前，应先加只读/日志层面的测试与记录：

1. 对同一实例窗口保存 association logits、activated probability、Hungarian 配对和旧 threshold action；
2. 记录 track ID 与 proposal index 的映射，避免后续重排误配；
3. 记录 `with_bank=False/True` 两条路径的 memory read/write、reactivation 和新 ID；
4. 为每个决策保存 legal action mask、question type、state digest、JEV mode 和 runtime；
5. 将 offline counterfactual rollout 的 future GT 与 online inference 输入严格分开；
6. 验证 OFF 输出与旧实现逐框、逐帧、逐 ID 一致，再开启任何学习型 gate。

## 结论

JEV 的第一处真正插入点是 `run_global_tracker_plus` 的阈值/新 ID分支，但必须先抽出 proposal、legal action、decision、commit 四层，并为 memory 路径建立同样的 contract。association head、detector、ReID 和 GMT Stage2 在此之前应冻结；任何性能结论都必须来自共享 checkpoint 和同一严格 online protocol。
