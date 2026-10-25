# V3 身份监督与推理决策对应关系

模型 Schema：`worldobject_identity_assignment_v3`。旧 W4/W5 checkpoint 不兼容。

## Tensor 契约

M 为当前历史对象数，N 为当前观测数，D 来自 GMT 冻结 Manifest，H 默认128。

| 输入 | 形状 | 来源 |
|---|---|---|
| history_features | M×D | 推理 Bank 中真实长期 ROI embedding，未初始化的历史缺失状态有明确置信度 |
| observation_features | N×D | 当前 GMT ROI；正式视觉方法不接受缺失特征 |
| pair_geometry | M×N×12 | 位移3、联合协方差对角3、可靠性/有效性、时间间隔和可见性等 |
| observation_context | N×4 | score、当前可见性、相机来源编码 |
| history_context | M×4 | age、missed、身份置信度、确认状态 |

完整无几何模式不生成空间支持、不做几何配对/门控/对象位置更新，pair_geometry 全零；
head-only 对照只将可学习头的几何输入清零。两者不能混称。

| 输出 | 形状 | 实际消费位置 |
|---|---|---|
| assignment_logits | M×N | 历史 ID 候选选择及联合选项收益 |
| null_logits | M | 相对当前分配证据决定 NULL |
| existence_logits | M | 选项收益、失踪生命周期 |
| birth_logits | N | 联合 Birth 可行性、未确认/新生状态 |
| update_gate | M×N | 关联成立后的空间和身份写入控制 |
| memory_delta | M×N×D | 修正正确 Global ID 的当前 ROI 候选并写回长期记忆 |
| corrected_memory | M×N×D | 仅在正确监督匹配上计算更新一致性 |

模型将历史 ROI、当前 ROI 和二者差异一起编码，不能对所有历史 ID 输出相同的检测 gate。
Camera ID 是来源上下文，模型不接收 scene_id、GT embedding 或数值 Global ID 特征。

## 目标与损失

| 损失项 | 监督 | 推理对应 |
|---|---|---|
| Identity Assignment CE | 每个检测选择历史对象或 Birth 类 | assignment_logits |
| Cross-view Shared Identity | 同人跨摄像头历史 ID 分布一致 | 联合对象归属 |
| False Merge | 不同人的历史对象概率重合惩罚 | 抑制同一 ID 解释不同人 |
| Birth BCE | 当前身份在合法历史中不存在 | birth_logits |
| NULL BCE | 已知历史身份当前无匹配 | null_logits 相对当前最强候选 |
| Existence BCE | 有观测或尚在合法保留期；超期且无观测为负 | existence_logits / Death |
| Update Gate BCE | 正确对象—检测对为正，错误对为负 | 匹配后的写入控制 |
| State Update Consistency | 正确匹配时修正后的 ROI 与当前真实 ROI 一致 | memory_delta / 持久 embedding |

没有真实3D GT，因此没有伪造3D监督损失。所有 GT 只进入 TRAIN Target Construction 或单独的后验评价。
目标构造只使用 instance_id/track_id；COCO annotation id 不能作为人物身份。
同帧同摄像头的重复身份、无效框、ignore/crowd、无身份标签会显式 mask；RGB 不被删除。
匹配一一对应，跨摄像头共享同一原始身份键；场景与 clip 边界重置局部监督映射。

## 因果状态训练

`predicted` 是默认训练历史：模型先从真实 Bank forward，再计算当前目标，最后用自己的身份决策更新同一个 Bank。
GT 侧的历史标签映射只用于 loss 和错误归因，不能被模型输入构造器读取。

`teacher_forced` 仅用于 TRAIN：先执行相同 forward、计算损失，随后用当前 TRAIN 对应纠正分配。
此前时刻的历史不能由当前/未来 GT 预先改写。推理模式拒绝 teacher decision。
状态以 NumPy detach 维护，是逐时刻监督加自主回放，不声称已经实现跨整个视频的可微 BPTT。

训练逐场景/逐时刻流式读取，不再把打乱后的独立 gate 样本作为主任务。默认 clip 长64。
训练对数超限、非有限 loss 或没有合法身份监督时明确失败，不能靠完整图像覆盖写出虚假训练 Gate。

## 选择、完整训练与 TEST Gate

准备的协议：18个 TRAIN fit 场景；4/8/12/16/20/24 为内部验证；按验证 Identity Assignment Loss
选择 epoch 数。固定配置后同 seed 重新初始化，使用全部24个 TRAIN、28,754个时刻最终训练。
默认4个选择 epoch 只是批准后预算，不代表已经训练或保证收敛；没有多 seed 队列。

`worldobject_identity_fulltrain_gate_v3` 记录模型配置、方法 ID、D、原始检测器契约、数据/源锁 SHA、
随机种子、训练预算、合法监督数量、逐场景覆盖和权重 SHA。旧 Schema、错维度、错误方法、
不完整场景、源代码不一致或权重 SHA 不符均拒绝 TEST 加载。
非学习的 rule-based 对照必须显式使用 `--rule-based`，不能冒充训练后的模型。

本轮只验证了合成梯度、分配影响和记忆读写；没有真实训练收敛曲线、TRAIN/TEST 精度或模型吞吐结论。
