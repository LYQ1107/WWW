# JEV online state schema

`gtr/modeling/jev_state.py` 将在线 GMT evidence、track/memory summary、view/time、disagreement 和预算计数编码为固定向量。前四列专门保留给阈值基线：current accept、reassociate、memory write、reactivation score。该编码器不接收 GT、future frame 或 evaluator 对象；未知字段会被忽略并应由 adapter 审计。

