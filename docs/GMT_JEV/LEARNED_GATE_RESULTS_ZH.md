# 单次学习式风险门控修订结果

旧 conflict 门控、Rule 完整 TRAIN 负向结果及旧零训练决策仍保留。当前是由 TRAIN 认证目标覆盖缺口支持的一次有界修订，不能用 ORACLE 见证直接宣称新模型有效。

当前阶段：REAL_SUPERVISION_PRECHECK_PENDING_NO_UPDATES；已记录 optimizer 更新总数：0（各头单独计数）。
三个头使用相同原生24维输入、候选/求解器、±0.05残差上界、原支持长度和提交规则。参数 MLP5119、Attention4898、CIC4962。
32场景帧的三个原生零残差合同均通过；此检查不是完整视频或独立验证。

| 完整冷启动 TRAIN16 | HOTA | ΔHOTA | AssA | CVIDF1 | IDSW | Frag | 完整图像FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| mlp | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| attention | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| cic | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

主要协议是完整native raw；原min50/stock另报，不按结果换协议。TRAIN16是开发集，普通MLP/Attention结果一并保留。单种子差值、训练loss和ORACLE均不能证明显著性或独立结构优势。
长期错误统计目前是回顾性描述量，accepted-reference N01/N10是在同一当前状态的局部比较，不能冒充整段干预因果传播。完整FPS来自GPU7三次真实图像推理，包括前端/关联/提交/最终stock；共享宿主机串行块存在时间负载差异。
完整目标仍待联合质量/效率、普通控制器优势、必要种子和合法合格独立视频。SCIENTIFIC_SUCCESS=false。
