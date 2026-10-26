# 阶段 A：公平同步身份核验

实际状态：`RUNNING_REGISTERED_COMPLETE_VIDEO_EXPERIMENTS`。

Protocol A 使用各自成功匹配的事件；Protocol B 使用与 GMT 共同 GT 事件。报告 TP/FP/FN、Precision、Recall、F1 和歧义掩码，不用总体 Accuracy 掩盖类不均衡。

真实测量、纠正与回归均以 results 中的完成报告为准。未测指标不使用历史 R2 数值代填。

本地输入镜像已全部校验。当前修订的两段真实工程预检再次 32/32 严格复现 C0/R2；六场完整 TRAIN 回放和已有 TEST 输出审计已启动。已完成场景与输入证明见 `v4_execution_attempts.json`、`v4_input_mirror_receipt.json`；完整正式分数仍待评分。
