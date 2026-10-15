# 首次输出分歧的原生状态配对核验

仅重跑TRAIN16前三个场景，两条轨迹均冷启动；不恢复旧JEV前缀、不启用旧JEV策略、不改变权重。前三场景预测分别与各自已保留完整轨迹严格一致。

| 当前关联前的共同输入 | 严格一致 |
|---|---|
| base_scores_SHA256 | True |
| before_derived_summary_SHA256 | True |
| before_native_state_SHA256 | True |
| candidate_ids | True |
| feature_SHA256 | True |
| lengths | True |

| 运行 | 离线目标类别 | 原始接受ID | 修正接受ID | 实际提交ID | 新生身份 |
|---|---|---|---|---|---|
| mlp | UNKNOWN_CURRENT | None | 25 | 25 | False |
| observe | UNKNOWN_CURRENT | None | None | 52 | True |

只有共同输入全部严格一致，才支持局部同状态比较。当前核验包含完整native状态/RNG、派生摘要、GTA原始分数、24维观测、候选及支持长度；记录了真实Bank/出生提交。
该行GT为UNKNOWN，不能把ID52改成ID25计为已认证N01，也不能说已证明纠正了身份错误。没有执行该事件单独开启/关闭的完整视频干预，不能把HOTA增益归因于此事件或宣称已测真实因果传播时间。
