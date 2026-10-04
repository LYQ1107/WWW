# GMT OFF decision contract

`gtr/modeling/gmt_decision_contract.py` 以纯函数记录当前 `run_global_tracker_plus` 的固定阈值语义：`not_mult_thresh=False` 时阈值乘轨迹长度，比较符号是严格 `traj_score > thresh`；失败分支创建新 ID。候选 reassociation 只返回一个未占用的第二候选，不能生成 ID。

