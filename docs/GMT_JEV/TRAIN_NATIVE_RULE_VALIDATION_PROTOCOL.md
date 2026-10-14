# 完整 TRAIN Rule 原生提交验证（执行前冻结）

TRAIN13 完整覆盖出现一次 TRAIN 时间桶局部 N01，前缀结论不能代表全部 TRAIN。先完成 TRAIN12/13/14 原始 GMT 覆盖补查；仅在三段 TRAIN 时间桶合计 Rule 局部 N01 大于 N10 时，使用同一冻结 Rule 对三段视频分别原生冷启动验证，不挑有利视频，不使用 TRAIN16 调参。

保持完整 GMT checkpoint、前端、GTA、候选、solver、Gallery/Bank 和原生提交不变。复用 collect 的只读离线认证观察器，Rule 实际运行并改变后续状态；GT 不进入 actor。每次提交与该次真实当前状态中的 raw GMT 分数所产生的 accepted MATCH 比较。拒绝 MATCH 后未执行的原始 Bank/新生身份结果记 UNKNOWN，不拼造反事实。唯一合法目标与既有历史认证、别名 masking、64帧时间桶规则不变。

N01 需要原始 accepted MATCH 的认证人物错误且 Rule 实际提交唯一目标。N10 需要原始 accepted MATCH 正确而 Rule 实际提交另一已认证人物，或相对于既有唯一目标新建身份；两类分别计数。其他 owner 未认证、Bank 反事实或别名语义不确定的改动记 UNKNOWN。新计数不复用 collect 中依赖实际提交的 GMT_correct 命名，完整保留其旧诊断统计并标明含义。

分别输出 TRAIN/heldout 时间桶与总计，记录可认证监督覆盖和风险门控覆盖，仅持久化小型统计与 provenance，不保留样本语料、不新增梯度更新。结果只证明当前 Rule 状态上的局部原生提交信号，不证明整段因果收益、HOTA 增益或独立验证。报告本身不会自动授权训练；仍需监督采样和公平训练协议核查。

验证在既有全视频覆盖队列完成并推送后逐段执行，使用空闲卡且不结束其他任务。总计仍受本轮12 GPU-wall-hours、512 MiB留存及目的盘30 GiB保留空间约束。禁止新 DEV、封存视频、TEST、Full24、下载数据或覆盖历史结果。
