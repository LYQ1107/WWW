# GMT Stage2 result metrics

更新时间：2026-10-05 02:03 UTC

这份文档只记录 Stage2 的可复核训练指标和 checkpoint 验证结果。checkpoint 本体、数据集、日志和事件文件保留在本机，不上传 GitHub。

## 训练复现实条件

- 训练阶段：GMT Stage2，单 GPU，Detectron2 world size `1`。
- 配置快照：`outputs/stage2_single_gpu/config.yaml`。
- 初始化权重：`/data1/liuyeqiang/WWW/outputs/stage1_single_gpu/model_16000.pth`。
- 数据集：`VISION_train`；`SOLVER.IMS_PER_BATCH=1`。
- 优化器：`ADAMW`；`BASE_LR=2e-5`；`WEIGHT_DECAY=1e-4`。
- 学习率：`WarmupCosineLR`，线性 warmup `1000` iter，`WARMUP_FACTOR=0.001`。
- 梯度裁剪：full-model L2 norm，阈值 `0.1`。
- 目标迭代：`20000`。
- 训练器日志报告总训练时间：`15:49:23`。

## 最终 checkpoint

| 字段 | 值 |
| --- | --- |
| 文件 | `/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth` |
| 字节数 | `638238241` |
| SHA256 | `cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8` |
| iteration | `20000` |
| scheduler_last_epoch | `20000` |
| global_iteration | `20000` |
| model keys | `411` |
| optimizer state | `PRESENT` |
| model finiteness | `PASS` |
| checkpoint reload | `PASS` |
| validation status | `PASS` |

权威验证文件：`/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/validations/model_20000.json`。

## 训练日志指标

`metrics.json` 共记录 `2000` 行；以下为最后一行（iteration `20000`）：

| 指标 | 最终值 |
| --- | ---: |
| `total_loss` | `1.2422004495747387` |
| `loss_asso` | `0.6270861029624939` |
| `loss_asso_smooth` | `0.0` |
| `loss_centernet_agn_neg` | `0.011058029718697071` |
| `loss_centernet_agn_pos` | `0.015652906149625778` |
| `loss_centernet_loc` | `0.12149083986878395` |
| `reid_loss` | `0.46364010870456696` |
| `lr` | `1.3632387953488315e-12` |
| `time` (s/iter) | `4.711133852601051` |
| `roi_head/num_bg_samples` | `505.8611111111111` |
| `roi_head/num_fg_samples` | `6.138888888888889` |

对照摘要：首条记录（iteration `19`）的 `total_loss` 为 `96.71438902243972`；最后 100 条记录的 `total_loss` 均值为 `0.8710850674601442`。

## 解释边界

上述是 Stage2 的训练 loss、学习率和 checkpoint 完整性指标，不是 MOT/COCO 的最终跟踪指标。正式 baseline、Oracle、JEV 策略和反事实评测仍须以 `research_final_v2` 中通过 manifest 的结果为准；当前研究流水线尚未完成全部正式评测。

