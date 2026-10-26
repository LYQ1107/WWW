# 阶段 A：公平同步身份核验

实际状态：`WAITING_FOR_NAS_TUNNEL_AND_MOUNT`。

Protocol A 使用各自成功匹配的事件；Protocol B 使用与 GMT 共同 GT 事件。报告 TP/FP/FN、Precision、Recall、F1 和歧义掩码，不用总体 Accuracy 掩盖类不均衡。

真实测量、纠正与回归均以 results 中的完成报告为准。未测指标不使用历史 R2 数值代填。

截至北京时间 2026-10-26 14:45，用户已确认 NAS 开机；本机检查仍显示反向隧道 `127.0.0.1:24557` 未监听、NAS 未挂载。冻结 R2 清单不可读，真实实验尚未启动；等待 NAS 端服务状态和 `v100.log`。恢复控制器仍在运行。
