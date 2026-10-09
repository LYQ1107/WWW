# Question redesign before Pilot

Frozen evidence: `reports/JEV_PHASE14/PHASE13_FROZEN_EVIDENCE.json`。

Phase XIII只有MATCH监督。动态六token的QuestionReader虽然使用当前检测、相机元数据、全局/本相机/其他相机历史与平均竞争证据，但没有独立Q2/Q3标签。Fixed优于Full不能证明这些证据无用；同前缀冻结权重诊断与实际多任务Pilot分别记录。

新模型共享一个StateEncoder和WHO的OptionReader表示。WHO选择合法ID；AVAILABILITY输出当前合法答案集合是否有正确身份的类型化概率；TRUST按当前检测/身份对输出已有观察历史的纯净概率。MultiQuestion用三种类型重新读取同一state，参数共享。Full多任务控制用同一个WHO query接普通不同heads；Fixed仅保留固定任务query；Set、MOTIP-style、普通MLP获得相同视觉/元数据、Q2/Q3标签、联合求解器和预算。

Q2/Q3在TRAIN自然native过去观察上认证；UNKNOWN不当负例。Q3是观察历史纯净/污染问题，不是未来WRITE/KEEP收益，更不能宣称已训练完整MATCH–REACT–MEMORY生命周期。

候选移除只改变版本化训练副本的合法Option集合，监督MATCH DEFER。新增同候选数presence对照，移除同数量认证错误身份而保留正确身份，避免仅靠K变化解题。评估保留未干预自然样本，并单列自然/人工和难度范围。

尚未完成的真实在线和三种子结果不填数字。`PILOT_PROTOCOL.json`在训练前冻结，主标准和历史门槛不改。
