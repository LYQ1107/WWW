# JEV 训练入口

`reproduction_tools/train_jev.py` 是离线 decision-policy trainer。它先严格验证 JSONL contract，再从 `state.feature_vector` 读取当前状态；`action_outcomes`、`best_actions` 和 future GT 不进入模型输入。sequence split 在训练前完成，避免相邻帧泄漏。

可比较的 `--model` 包括 `fixed_threshold`、`jev`、`fixed_slot_mlp`、`independent_mlp`、`shared_heads`、`logistic`、`global_threshold` 和 `state_threshold`。正式数据必须来自冻结 GMT Stage2 的真实 counterfactual rollout；当前 CPU smoke test 只验证训练管线，不构成研究结果。

训练产生的 `model.pth` 保存了 `model_name/state_dim/hidden_dim` 等重建元数据；
GMT runtime 只接受显式的、可加载且维度匹配的 checkpoint，不会随机初始化控制器。
