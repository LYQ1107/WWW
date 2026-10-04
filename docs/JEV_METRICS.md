# JEV 决策指标

`reproduction_tools/jev_metrics.py` 对齐 semantic legal action 名称计算 NLL、Brier、ECE 和 risk-coverage，避免不同 action 排序造成虚假比较。它只评估 policy output；HOTA/IDF1/IDSW/Frag 等 tracking 指标仍由冻结 GMT 的正式 evaluator 计算。

