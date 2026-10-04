# 正式 GMT inference

`reproduction_tools/run_formal_inference.py` 使用当前 `/data1/liuyeqiang/WWW` 仓库的 `test_net.py`，要求显式提供 Stage2 checkpoint、配置和新的输出目录；它会记录 checkpoint/config/result SHA256 和命令，拒绝覆盖已有结果。正式评测必须在 Stage2 最终 checkpoint 和 OFF/SHADOW gate 通过后执行。

