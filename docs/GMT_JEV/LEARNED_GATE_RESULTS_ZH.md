# 单次学习式风险门控修订结果

旧 conflict 门控、Rule 完整 TRAIN 负向结果及旧零训练决策仍保留。当前是由 TRAIN 认证目标覆盖缺口支持的一次有界修订，不能用 ORACLE 见证直接宣称新模型有效。

当前阶段：COMPLETE；已记录 optimizer 更新总数：900（各头单独计数）。
三个头使用相同原生24维输入、候选/求解器、±0.05残差上界、原支持长度和提交规则。参数 MLP5119、Attention4898、CIC4962。
32场景帧的三个原生零残差合同均通过；此检查不是完整视频或独立验证。

认证监督前置检查：PASS_NATIVE_ZERO_AND_SUPERVISION_CONTRACT；临时缓冲峰值 14.559 MiB。
TRAIN与时间留出桶计数：

```json
{
  "TEMPORAL_HELDOUT": {
    "E1_certified_original_wrong": 30,
    "E2_certified_original_correct": 5336,
    "UNKNOWN_or_no_active_target_rows": 6841,
    "certified_unique_rows": 5366,
    "masked_E1_without_joint_capped_assignment_witness": 23,
    "masked_capped_threshold_margin_unreachable": 5,
    "masked_rows": 6869,
    "rows": 12207,
    "supervised_E1_witness_rows": 2,
    "supervised_preserve_rows": 5336
  },
  "TRAIN": {
    "E1_certified_original_wrong": 97,
    "E2_certified_original_correct": 26200,
    "UNKNOWN_or_no_active_target_rows": 25270,
    "certified_unique_rows": 26297,
    "masked_E1_without_joint_capped_assignment_witness": 71,
    "masked_capped_threshold_margin_unreachable": 9,
    "masked_rows": 25350,
    "rows": 51567,
    "supervised_E1_witness_rows": 17,
    "supervised_preserve_rows": 26200
  }
}
```
时间留出桶的矩阵不参与梯度更新；GT只作离线认证，UNKNOWN、不唯一、阈值不可达及无联合有界纠错见证的错误行被mask。原生完整矩阵仍参与训练求解，UNKNOWN候选竞争不能忽略。
时间桶并非严格标签隔离：原生轨迹与离线历史认证跨桶连续，较早留出桶的历史GT可参与随后TRAIN目标的历史认证。该桶仅用于诊断，不报告为独立或严格标签隔离验证；当前完整闭环仅在已暴露TRAIN16开发视频上评测。

| 头 | 固定终点 | 新增参数 | 训练样本顺序 SHA |
|---|---:|---:|---|
| attention | 300 | 4898 | 88199ddda16272a4a4139e233818d005141b34ea7960c54fcad6a2421525bf82 |
| cic | 300 | 4962 | 88199ddda16272a4a4139e233818d005141b34ea7960c54fcad6a2421525bf82 |
| mlp | 300 | 5119 | 88199ddda16272a4a4139e233818d005141b34ea7960c54fcad6a2421525bf82 |

AdamW lr0.001、weight_decay0.0001、FP32、clip1、seed20261009；结构化 assignment hinge + threshold hinge +10×preserve +residual penalty。训练loss不证明完整闭环质量。

| 完整冷启动 TRAIN16 | HOTA | ΔHOTA | AssA | CVIDF1 | IDSW | Frag | 完整图像FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| mlp | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| attention | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| cic | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

主要协议是完整native raw；原min50/stock另报，不按结果换协议。TRAIN16是开发集，普通MLP/Attention结果一并保留。单种子差值、训练loss和ORACLE均不能证明显著性或独立结构优势。
长期错误统计目前是回顾性描述量，accepted-reference N01/N10是在同一当前状态的局部比较，不能冒充整段干预因果传播。完整FPS来自GPU7三次真实图像推理，包括前端/关联/提交/最终stock；共享宿主机串行块存在时间负载差异。
完整目标仍待联合质量/效率、普通控制器优势、必要种子和合法合格独立视频。SCIENTIFIC_SUCCESS=false。
