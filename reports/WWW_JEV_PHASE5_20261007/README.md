# Phase V 可核对证据

完整报告在 ../../docs/WWW_JEV_PHASE5_DIAGNOSTICS_20261007.md。

本目录包含所有冻结标签、修正 MATCH 标签、十次跟踪预测/决策、训练与校准结果、checkpoint、完整审计与日志。JSON/JSONL 的 gzip 文件使用固定 mtime=0；原文件 SHA 与压缩文件 SHA 对照见 ARCHIVE_MANIFEST.json。

核对目录内文件：

```bash
sha256sum -c SHA256SUMS
```

核对压缩预测的原文件 SHA，例如：

```bash
gzip -dc minimal_tracking/B2/tracking_predictions/jev.json.gz | sha256sum
```

结果应与该条件 result.json 的 predictions_sha256 一致。B2/B2_repeat 的预测、动作与完整在线 feature/context 日志 SHA 相同。模型加载与动作检查在 MLP_POLICY_SANITY.json；公平对照在 CORRECTED_THREE_WAY_COMPARISON.json。

在原项目环境重新运行 B2（必须选择新的输出目录）：

```bash
env CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/liuyeqiang/anaconda3/envs/GMT/bin/python /data1/liuyeqiang/WWW_jev_phase5/reproduction_tools/run_jev_phase5_ablation.py --variant A1 --checkpoint /data1/liuyeqiang/WWW_jev_phase5/reports/WWW_JEV_PHASE5_20261007/minimal_training/B2/calibration/model_calibrated.pth --output /home/liuyeqiang/WWW_jev_phase5_runtime/20261007/reviewer_B2_fresh
```

该命令仍需原项目的 detectron2/GMT 环境、冻结 GMT model_20000、TRAIN 数据与感知缓存。外部输入的固定 SHA 见 PHASE5_BASELINE_LOCK.json；这些大文件没有上传。正式 TEST 未运行；eval_dataset 内的 test 名称只是 TRAIN 诊断子集的评测格式。

新 counterfactual builder CLI 默认使用 cache0_annotation1 坐标。函数层 legacy 默认保留给冻结复现；新程序调用应显式传入 gt_coordinate_contract="cache0_annotation1"。反事实 utility 的 prefix-reset / 固定 OFF action-category continuation 限制仍在报告中保留。
