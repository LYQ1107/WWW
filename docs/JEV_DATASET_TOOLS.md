# JEV offline dataset tools

`reproduction_tools/jev_dataset_tools.py` 只属于 counterfactual labeler：它按同一状态内的 H-step utility 生成 best-action 和 soft target，记录温度、tie-break、horizon 和 state digest，并按 sequence 做 train/validation/test split。

当前工具不会伪造 GMT rollout；`action_outcomes` 必须由之后接入冻结 Stage2 的真实 branch runner 提供。只有那一步完成后，JSONL 才能用于 JEV 训练。严格 online consumer 仍必须使用 `validate_record(..., allow_future_gt=False)` 并拒绝 `uses_future_gt=true`。

