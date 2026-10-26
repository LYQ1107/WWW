# 阶段 A：公平同步身份核验

实际状态：`STAGING_BYTE_VERIFIED_LOCAL_INPUTS`。

Protocol A 使用各自成功匹配的事件；Protocol B 使用与 GMT 共同 GT 事件。报告 TP/FP/FN、Precision、Recall、F1 和歧义掩码，不用总体 Accuracy 掩盖类不均衡。

真实测量、纠正与回归均以 results 中的完成报告为准。未测指标不使用历史 R2 数值代填。
