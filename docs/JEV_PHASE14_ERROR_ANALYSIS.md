# Phase XIV 身份错误归因（P1）

Phase XIII 固定证据：`reports/JEV_PHASE14/PHASE13_FROZEN_EVIDENCE.json`。本报告只读既有预测与 actor journals，没有重新运行视频，也没有训练。

39 组真实完整视频日志（development17/18/19，Full、Fixed Question、Set、Cosine、Full24k）全部完成。逐帧使用原始 TrackEval dataset preprocessing 和 CLEAR 的上一帧连续性优先匹配，重建 IDSW 与原指标逐序列完全一致。

| 方法 | 三视频 IDSW / 种子 | 新混合事件 / 种子 | 可认证 false split / 种子 | false birth / 种子 | UNKNOWN 抢占已有正确候选 / 种子 |
|---|---:|---:|---:|---:|---:|
|formal/cosine|82.000|16.000|20.000|21.000|3331.000|
|formal/fixed_question|105.000|32.333|2.667|4.333|764.667|
|formal/full|118.667|33.667|5.000|6.667|1021.000|
|formal/set_transformer|158.333|36.667|4.333|7.667|886.667|
|onpolicy/full|143.667|33.333|4.667|7.000|786.333|

Full 相对 Fixed Question 的额外41次 IDSW（三种子合计）中，纯净旧碎片切换/恢复增加38次，UNKNOWN/污染旧身份变化减少1次，错误已有身份合并增加4次，split出生增加2次，其他false birth减少2次。它不是“全部因为新出生变多”的结果，也不能把污染轨迹内切换自动认证为新错误合并。

Full20k 的356次 IDSW中：纯净碎片切换/恢复145、UNKNOWN或污染旧身份变化148、错误已有身份合并51、错误stale恢复1、可认证split出生8、其他false birth3。

这些是不同策略自身状态轨迹的描述性分解，不是固定前缀的因果对照。已有污染状态下的选择仍为 UNKNOWN；可确认的历史跨GT混合另外统计，不能把所有UNKNOWN当错例。

跨摄像头 mismatch 计数是最近40帧内、逐检测观察到的跨视角 ID 不一致；它不能代替官方CVIDF1/CVMA。持续区间是连续观察到的 taxonomy 事件，间隔和结尾删失，不是反事实恢复时间。

训练/部署差异：旧CE/Brier只归一化认证选项，旧structured loss移除UNKNOWN竞争者；native部署保留全部合法候选。新实验将保留同求解器并检验支持该差异的风险训练，不以GT屏蔽在线候选。

`ERROR_ATTRIBUTION.json`含逐案例计数、精确IDSW重建、身份持续区间、示例与所有源SHA。完整事件流在服务器，只上传紧凑结果。典型事件可以根据保存的原actor source/checkpoint从bootstrap复现；不把已删除的临时resume槽冒称为已有永久快照。
