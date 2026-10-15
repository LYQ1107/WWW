# 学习式风险门控有界 smoke（执行前冻结）

三段完整 TRAIN 对照中，冻结 Rule 的主要 HOTA 均下降；合并 HOTA −0.008859，IDSW 477→533。严格原生 accepted-reference 子集 N01/N10=0/1。保留这些失败结果，不把它们改写为正向。受限 ORACLE 指出：TRAIN 时间桶97条认证错误中，原手工门控仅开放11条目标边；在不改候选/求解器和±0.05残差幅度的诊断中，移除门控可提供17条 E1 目标接受见证、3次严格 N01。这不是模型表达能力或在线增益证明。

唯一修订：小型 MLP/Attention/CIC 三头统一使用 learned risk mode，让同样的 sigmoid 门控判断全部原生合法边；不再由手工 margin、其他相机支持数或余弦差硬排除。这些量继续作为相同24维可观察输入。旧 conflict mode 是默认值，现有 Rule/Threshold/Confidence 及旧结果不改。模块仅加原 GTA 后残差，幅度仍≤0.05，支持长度缩放、原匹配约束和 Gallery/Bank 提交不变。零残差仍应严格等于 GMT。

先完成监督与提交契约 precheck，才允许一次300-step smoke：TRAIN12/13/14，原64帧时间桶隔离heldout，原3次/80%历史认证与年轻别名 UNKNOWN 规则不变。GT只用于监督，不进入推理。原 GMT 原生运行，即时采样到≤32MiB的临时RAM缓冲，统计每段、E1/E2/UNKNOWN、门控与阈值可达性、采样覆盖和masked比例；不保存训练语料、不读取16梯度、不触碰新DEV/封存/TEST。

三头约5k参数，同一数据、固定采样顺序和20261009种子，同一AdamW(lr=0.001, weight_decay=0.0001)、FP32、梯度裁剪1.0和300步终点。只保存300步checkpoint，不按TRAIN16选择step。L_assoc使用同一当前候选矩阵上的约束匹配结构化hinge及原接受阈值hinge；目标必须认证唯一且在残差上界下可通过原接受阈值，不能强迫非法/不可达目标。匹配margin为0.005平均分数单位。L_preserve约束原本正确认证行的残差，L_res约束整体残差，系数分别10和1。GT_UNKNOWN/别名行不得伪造目标；统计其耦合与未监督覆盖。

必须先验证：默认 conflict 不改变旧行为；learned mode零残差分数/原生提交/Gallery/Bank一致；输出只作用于当前合法候选且满足幅度；监督采样不混入heldout；非零可达监督有有限梯度；冻结GMT无梯度。若precheck不满足，不启动300步。若满足，再运行三头相同预算的smoke与固定终点完整在线比较，不扩大到1000/20k，不再搜索新架构。保留普通MLP结果，不强求JEV优势。

这是一次由TRAIN覆盖缺口支持的门控修订，原Rule失败与accepted-reference训练信号失败仍成立，不能用ORACLE直接授予科学成功。计划与实际训练分开记录。整个任务累计仍≤12 GPU-wall-hours、新留存≤512MiB、home≥30GiB；有界检查和可能的训练不是独立验证，所有论文级完成条件保持不变。

执行细节在采样前冻结：完整矩阵按E1和preserve两个层各≤14MiB的SHA256(key,seed)优先级留存，计入tensor storage和Python容器，缓冲总量<32MiB；不截断候选或行。监督目标列重复的行全部mask。目标在±0.05上界下连0.105（原0.1阈值+0.005 margin）都不可达的行mask；错误行还须在联合认证目标的受限重评分见证中真实通过同一native assignment。见证只用于离线监督资格，不改变实际GMT提交，不证明模型可达或在线收益。UNKNOWN行仍留在完整矩阵中，参与求解竞争但无伪造GT目标。

结构化hinge用同一矩形一对一约束：参考解固定认证监督目标，剩余行列做Hungarian；loss-augmented解在认证目标边减去0.005×原支持长度，再用相同Hungarian。两解的可微分数差加任务margin，经relu并按认证目标支持总量归一化。另加目标接受阈值hinge、原本正确行残差平方preserve及全矩阵残差平方正则。300步在E1/preserve矩阵间交替，固定随机顺序，三头完全相同；不持久化样本，也不把训练loss当成在线性能证据。
