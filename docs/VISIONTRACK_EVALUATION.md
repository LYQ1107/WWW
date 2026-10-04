# VisionTrack 正式评测路径

1. `test_net.py` 保存 `coco_instances_results.json`；VisionTrack 不再误走旧的 MOT17 分支。
2. `prepare_visiontrack_predictions.py` 将预测转换到隔离的 TrackEval GT/tracker 目录，并为 cross-view 原始转换器准备 XYXY 输入。
3. `evaluate_visiontrack.py` 使用 `SEQ_INFO` 和下载数据的 GT，运行仓库 TrackEval 的 HOTA/CLEAR/Identity；评测返回值和 NumPy 指标会被显式转换为可审计 JSON。

转换器只创建 GT symlink，不改 GT 行；每次输出目录都拒绝覆盖，并保存 annotation/prediction SHA256。当前原始 test GT 中有若干同一帧重复 track ID，严格 TrackEval 会拒绝这些行；只有显式传入 `--allow-duplicate-gt` 才运行审计模式，GT 仍保持原样，报告会记录重复行数量。正式 Stage2 checkpoint、OFF/SHADOW 等价门和预测完成后，才能执行这条链。

空预测的 44 序列 smoke test 已通过：HOTA/IDF1/MOTA 为 0，GT 检测数为 580,178；这只是评测链路和 GT 处理的回归测试，不是 GMT 结果。

跨视角结果使用 `reproduction_tools/evaluate_crossview_visiontrack.py`：
`sequential` 虚拟时间轴对应 CVIDF1，`interleaved` 虚拟时间轴对应 CVMA。
当前 Python 复现报告明确记为 `CVIDF1 = TrackEval Identity IDF1×100`、
`CVMA = TrackEval CLEAR MOTA×100`，并同时保存 raw 值；官方 MATLAB
R2020a/CV kit 结果只有在合法 MATLAB 与 Engine 可用后再作为独立核对，不能
用 Python 结果冒充官方 MATLAB 执行记录。
