# Phase XVIII — Native owner / candidate evidence report

本轮真实执行完成，学习资格仍 NO_GO；MASTER 科研目标没有完成。冻结 Phase XVII d44c30d 与旧负结果，只增加新代码与诊断。用户已确认下一步追加 TRAIN02–09、预留10/11/15的数据审计与有界实验；本轮输出不作为追加数据上的结果。

## 完整轨迹揭示什么

严格授权 TRAIN12/13/16 共90,058个观测，5,598个owner错误/假拆分且正确owner在合法active MATCH池中的事件。14单列旧重复GT诊断。“严格授权TRAIN”指划分，不代表video16全部GT唯一。事件多并不等于独立、可安全学习的纠错标签。

|video|观测|AVAILABLE_ERROR|UNAVAILABLE_ERROR|
|---|---:|---:|---:|
|12|14291|233|1249|
|13|16877|118|946|
|16|58890|5247|4345|
|14|32293|54|1036|

“UNAVAILABLE”只指active MATCH；接受MATCH时并未构造反事实Bank候选，不能推成所有历史都不存在正确身份。当前纯净支持与最早不可变owner分别统计；owner由过去单GT≥3、覆盖≥0.8建立，污染后仍保留，不是完美GT归属证明。GT不进入actor。

完整余弦排名、正确候选覆盖、混合历史与NN独立行偏好见FULL_CENSUS_EVIDENCE_SEPARABILITY.json。video16有5,247个可用错误，其中4,905个所有正确anchored-owner候选历史均已混合，GT-free备选仅覆盖3,678个；不能以全池Recall接近100%推断可可靠判断owner。该审计是描述性证据，不证明信息论不可能，也不是学习策略表现。

## 实际干预与安全归因

64个时间冲突组固定采样；16组执行16基线+57备选，共73个完整原生H64分支，同时检查H16/32/64。每分支从相同完整native状态和RNG恢复，第一步候选/分数/合法性完全一致；BASE续帧ID与完整原轨迹一致，原轨迹恢复后IDs、向量与raw memory哈希逐步一致。替代动作只依事前NN/余弦规则，不按GT选择。

冻结XV native接口实际只有一次MATCH Hungarian，然后残余Bank/REACT、NEW和Gallery写入。新探针锁定一个合法ref或private DEFER，再用同一个lawful_choice求解余行并检查容量。没有第二轮MATCH评分校验；此处是因果动作探针，不是已经可部署的学习controller。

|组|event key|划分|类别|严格安全纠错变体|有害变体|
|---:|---|---|---|---|---|
|0|[13, 363, 0, 6]|RESERVED|AVAILABLE_ERROR|ALT0|ALT1|
|1|[16, 832, 0, 9]|RESERVED|AVAILABLE_ERROR|—|ALT1,ALT3|
|2|[12, 776, 1, 5]|RESERVED|AVAILABLE_ERROR|ALT0|ALT1,ALT2|
|3|[16, 16, 1, 12]|TRAIN|AVAILABLE_ERROR|ALT0|ALT1,ALT2,ALT3|
|4|[12, 256, 0, 0]|TRAIN|AVAILABLE_ERROR|ALT0|ALT1,ALT2,ALT3|
|5|[16, 192, 0, 26]|TRAIN|AVAILABLE_ERROR|—|ALT1,ALT2,ALT3|
|6|[16, 117, 1, 17]|TRAIN|AVAILABLE_ERROR|—|ALT0,ALT1,ALT2,ALT3|
|7|[16, 448, 0, 7]|TRAIN|AVAILABLE_ERROR|—|ALT0,ALT1,ALT2,ALT3|
|8|[12, 1088, 1, 3]|TRAIN|AVAILABLE_ERROR|—|ALT0,ALT1,ALT2,ALT3|
|9|[12, 192, 1, 5]|TRAIN|AVAILABLE_ERROR|—|ALT0,ALT1,ALT2,ALT3|
|10|[12, 1041, 1, 3]|TRAIN|AVAILABLE_ERROR|—|ALT0,ALT1,ALT2|
|11|[12, 619, 1, 4]|TRAIN|AVAILABLE_ERROR|—|ALT0,ALT1,ALT2|
|12|[12, 640, 0, 3]|RESERVED|CORRECT_CONTROL|—|ALT0,ALT1,ALT2|
|13|[13, 128, 0, 0]|TRAIN|CORRECT_CONTROL|—|ALT0,ALT1,ALT2,ALT3|
|14|[13, 1216, 0, 4]|TRAIN|CORRECT_CONTROL|—|ALT0,ALT1,ALT2|
|15|[16, 1024, 1, 24]|TRAIN|CORRECT_CONTROL|—|ALT0,ALT1,ALT2,ALT3|

共有28个动作含真实纠错，50个动作存在至少一项有害后果；两者可重叠。4个动作、4个时间代表通过严格局部安全；owner-only安全诊断7个不能替代严格CLEAR资格。纠错必须是已知错误→已知正确，或有事前anchored支持的目标假拆分恢复；错误→UNKNOWN不算纠错。逐行N01/N10、UNKNOWN、污染/拆分、其他原MATCH行重配、每GT-camera错误传播和CLEAR差异见NATIVE_PAIRED_OWNER_OUTCOMES.json。

安全判定同时检查所有受影响行、三个H：反改错、正确覆盖丢失、新混合/假拆分、更多错误观测、任一GT-camera更长错误段、更多严格IDSW均否决。错误传播按真实检测观测计数，缺帧与不可见区间删失，不能声称完整不可见期间已知错误持续时间。

## 为何仍不训练

在查看主干预结果前冻结的跨时间owner/conflict复核，将16组并为6个保守单位；2个单位同时含TRAIN与RESERVED，不能贡献独立预留验证。结果后未重采样。真正合格安全TRAIN=0、RESERVED=1，少于原预注册8+3。邻帧、相关备选及三个重叠H均不是独立样本，没有显著性/泛化声明。

video16首轮严格TrackEval在View2 timestep184发现重复GT ID44而中止。12/13成功结果保留，16已完成的两组未重复，余四组按相同状态/动作恢复执行。受重复GT影响的过去前缀或未来窗口，CLEAR置null；不删除/重编号GT，也未调用permissive scorer。16个动作无法获得全部H的严格CLEAR资格。原始失败队列、日志、部分结果和修复actor均保留，工程失败不解释成算法正负收益。

## 结论与下一步

冻结困难前缀确实未覆盖完整轨迹中大量候选可用错误，且部分原生合法动作能安全纠错；但许多局部纠错产生联合关联副作用，监督独立性和安全性仍不合格。当前结果没有证明JEV结构优于同证据MLP/Attention，更没有证明未训练的新架构可泛化。

新训练、同证据学习基线、三生命周期新共享模型、完整冷启动新policy、部署速度和独立controller测试全部NOT_RUN；新HOTA/AssA/IDF1/MOTA为null。既有Phase XV/XVII反复DEV指标不是本轮独立结果。20/21/22、官方TEST、Full24均未启动。下一方案和8GiB资源上限见PHASE18_NEXT_SCOPE_PROPOSAL.md，追加范围先做历史暴露/GT/缓存审计，失败不换有利序列。

代码：jev_phase18_census/select/independence/contract/intervene/metrics/outcomes/separability/finalize及native constraint；持续研究controller已支持Phase XVIII。实际actor版本、冻结checkpoint SHA、输入/输出哈希由各JSON绑定。GitHub只推紧凑代码、报告、统计与manifest；视频、权重、原生状态和逐帧日志保留服务器。
