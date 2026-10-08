# Phase X 三模型实际实验与停止门槛

原生状态、H32和v3数据门槛均通过，随后实际训练了三个约34K参数模型。固定TRAIN12/13/14/16的12行、8个原组；相同state64/evidence12、完整合法候选、TRAIN-only归一化、双头、seed20261008、AdamW0.001/weight_decay1e-4/clip5。每次诊断每个架构均为500步，必要时共同扩展到预声明的2000步；没有给JEV独占标签或预算。

| 目标（共同2000步） | MLP已确认Rank-1 | DeepSets已确认Rank-1 | JEV已确认Rank-1 |
|---|---:|---:|---:|
| conditional_correctness_CE | 50.64% | 86.54% | 86.54% |
| known_edge_binary_CE_diagnostic | 50.64% | 86.54% | 86.54% |
| preregistered_joint_diagnostic | 68.59% | 100.00% | 100.00% |

正确性CE原实验与二元CE诊断均保留。联合目标使用原计划的正确性CE和已执行、未censor的H32偏好排序，任务均值1:1、部署双头和；未执行Q为NaN且不进入梯度。GT未知不标负，也不计正确。所有模型数值有限、候选排列选择不变、训练与TorchScript归一化一致；CUDA融合运行问题已在正式学习之前通过统一关闭融合修复。早期tiny历史缩进错误及其500步checkpoint也保留。

全部三模型必须达到95%的完整合法集合已确认Rank-1。联合目标下MLP仍为68.59%，其中31.41%的加权选择属于身份锚点未知的候选；已知支持上的NLL和已执行对比拟合不能替代完整合法选择的认证。这不是证明这些未知候选GT错误，也不是架构能力上限的证明。

因此正式三seed/20–50–100epoch、validation选择/校准、结构消融和heldout完整闭环均NOT_RUN。Tiny模型是实际训练的局部诊断，不能当作正式模型结果。每个checkpoint的服务器绝对路径、SHA、源码commit、样本选择SHA和完整原始结果来源在MLP_RESULTS/DEEPSETS_RESULTS/JEV_RESULTS、TINY_OVERFIT和SUPERVISION_ABLATION中记录；大文件未上传GitHub。
