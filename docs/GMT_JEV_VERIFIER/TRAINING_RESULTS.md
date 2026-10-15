# 真实训练记录

Q1阶段：COMPLETE_Q1_DIAGNOSTIC_ONLY。Typed/ordinary/Q2/GMT/原MLP新增更新均为0。

Q1若训练只使用TRAIN12/13非留出桶，固定300步；TRAIN14只作Q1开发诊断。当前GT不作为历史特征，UNKNOWN仅为过去证书不充分的三分类标签，不是动作伪标签。

完整逐25步loss/梯度、权重SHA、类别支持与混淆矩阵见 TRAINING_METRICS.json / Q1_RELIABILITY_DIAGNOSTIC.json。
Q1分类成绩不表示已改善HOTA、通过Q2机制或实现因果身份保护。

| Q1视频14诊断类 | Precision | Recall | F1 | 真值样本 |
|---|---:|---:|---:|---:|
| RELIABLE | 12.9032% | 2.5000% | 4.1885% | 320 |
| CONFLICTED | 24.0076% | 88.1944% | 37.7415% | 144 |
| UNKNOWN | 92.2830% | 76.6355% | 83.7345% | 749 |

预设softmax≥0.9的CONFLICTED预测数：0；固定外观代理预测数：0。没有按结果调低阈值。
Precision/Recall按采样诊断行计算，不是独立人物统计显著性或已校准置信度。历史分类不等于在线保护；该头从未参与native commit。
