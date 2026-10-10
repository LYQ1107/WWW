# WWW / JEV Phase XVI 最终研究报告

执行完成；科研结论 **NO_GO**。完成 P0/P1/P2/P3、真实图像完整 DEV、官方 MATLAB 和实际延迟。没有合格依据启动新读取器、Tiny/Pilot、三种子20k或 matched4k；未运行指标为 null，既有 XV 失败与权重保持原样。

冻结起点 `3e16cbf914d70bbde0d999faabc85afc9b5ae43f`。分支 `jev/www-jev-phase16-evidence-recovery-20261011`。V1 是冻结证据读取，V2 是冻结权重安全写入；V3 学习版因前置资格失败未启动。数据为 TRAIN12/13/14/16 与已反复使用的 DEV17/18/19。未使用20/21/22的视频缓存、图像、所选决策标签或评测；现有共享TRAIN标注train.json按标准loader解析并哈希，逐视频观测和标签提取限定在上述TRAIN/DEV。official TEST 未读取，Full24未执行。Stage1 已接触全部24个 TRAIN，不能声称全系统未见泛化。

## 证据与读取

P0 实际重放19个完整轨迹和72个 H32 分支。严格 TRAIN 的85793/85852条已知查询存在历史人物内容，但这不保证对应 Global ID 的纯净 owner。原生 REACT 仅在 MATCH DEFER 和真实合法 Bank 状态下调用前三槽最大余弦与0.75门槛；不是学习式 Q5，也不能将 DEFER 当安全拒绝。

P1 七个视频、五种四槽读取；冻结同一个 XV V3 seed20261009 LAST1500 权重，零训练更新，原始 logits、所有实际 ID 和最终原始记忆精确不变。A支持召回99.389%，剩余最大空间0.611pp，低于冻结+2pp门槛。这是协议与基线饱和的限制，不能降低门槛追认成功。C增益约0.145pp、D/E约0.076pp，全部不合格。局部原型改善部分影子错激活不能证明原生安全收益；A局部 token 纯净性未测量，报告null，不以初始化0作对比。

## 安全写入与因果归因

五个 GT-free 规则在相同实际 raw commits 上做影子比较。bounded_quarantine 在结果前选定；Gallery、hits、Bank、ID输出按各组真实原生动作更新并保留，原始观测不删除。独立匹配视图控制 WHO历史与 Q4前驱，CONFIRMED 只表示规则一致性。

24个原始前缀×四组=96条真实变异未来，H8/H16/H32全部保留，八个预定前缀额外H64。严格前缀18个，TRAIN14六个因重复GT只作诊断。A/B/C/D起始前缀、RNG、首次原始输入和原始分数相同；A全部H32 ID与XV原分支精确相等。C/D的E读取只干预一次目标决策，是失败读取器负对照，不是部署扩展。后续状态从未同步。

预选记忆规则门槛：**NO_GO_SAFE_MEMORY**。严格有害prefix/H检查 18 个，安全收益独立cluster 2 个（要求≥8）。所有有害路径、身份错误 episode 和交互效应保留在 `MATCH_VS_MEMORY_CAUSAL_FACTORIAL.json`。窗口求和存在重叠，不能当全视频或独立样本统计。

下面为18个严格前缀的H32描述性窗口求和，**窗口重叠，不是独立事件数或完整视频指标**。

|组|Future CLEAR IDSW|新增混合|错误owner观测|UNKNOWN owner观测|
|---|---:|---:|---:|---:|
|A|63|36|255|552|
|B|96|62|246|516|
|C|61|36|255|552|
|D|94|62|246|516|

E的首次决策干预在这组前缀仅改变一条严格目标提交；其IDSW小幅变化没有产生合格的必要身份纠错标签。记忆组的混合和IDSW恶化，说明隔离匹配证据不能自动保证提交身份正确。最早反例H32短期改善仍在H64增加混合3次和false_split_birth1次，按预定规则否决。没有据此追改参数或挑选序列。

认证owner为过去首次单GT、至少三条、已知覆盖≥80%，建立后不因混合而取消。错误转UNKNOWN不算纠错。传播只统计实际检测到的错误帧，观测断点与horizon/video end删失；额外按每个GT/camera比较最长错误episode，避免总量掩盖个别人恶化。

标签资格：安全恢复 0 cluster，不安全恢复 0，reserved必要纠错 0，结论 **NO_GO_RECOVERY_LABEL_ELIGIBILITY**。不制造缺失恢复动作标签，不启动第三个学习版本。

这些恢复标签计数专指24个冻结前缀的目标首次实际动作与安全后果。大量原有正常关联监督和记忆规则的有害后果分别保留；后者不能冒充新JEV恢复动作标签。这是本轮有界资格审计，不是对所有可能TRAIN动作作穷尽搜索。

## 完整实际图像与官方指标

下面仅是预选安全记忆规则的冻结单种子诊断，使用实际图像→Stage1 VFCE→JEV→native commit→updated memory。完整原始预测不筛短轨、不改GT、不重编号。

|指标|XV冻结参考|Phase XVI安全记忆诊断|
|---|---:|---:|
|HOTA|74.188270|61.764421|
|AssA|74.890708|52.052180|
|IDF1|90.521042|71.026365|
|IDSW|147.0|1494.0|
|MOTA|89.659419|86.258640|
|Frag|303.0|306.0|
|CVIDF1|90.285890|67.476069|
|CVMA|86.781450|72.580118|

各视频、认证覆盖、wrong-owner持续帧数、混合/出生、raw predictions SHA 在 ONLINE_VALIDATION.json 中。CVIDF1/CVMA 使用未修改官方 MATLAB、求和场景计数；完整toolkit哈希与历史一致。当前仅seed20261009，三种子新模型最差/均值均null。历史原关联最差824、XV V3的147完整保留，不能把单种子诊断当三种子恢复证明。

完整冷启动结果明确否决该规则：三个视频IDSW为320/431/743，合并1494，远高于XV参考147；HOTA、AssA和官方跨相机指标均严重下降。前缀实验共同保留旧历史，完整视频从冷启动使用新接纳规则，二者范围不同。这提示冷启动的隔离/fallback与改变后的WHO/Q4输入需要重新取得可靠资格；当前不能继续训练或部署该规则。主认证owner/false_split统计使用≥3与≥80%门槛，旧single-GT标签只作为明确标注的补充诊断。

## 实际运行成本

真实 DEV17前256双摄像头场景、三次重新启动、八帧warmup；无GT/evaluator/journal。场景FPS为 3.246, 3.282, 3.247，Stage2 p95为 56.645, 56.288, 56.189 ms。共享GPU情况、同步开销和组件计时全部披露。完整视频无observer FPS未测量，null。附加序列化状态每ID最多六条外观向量，派生归一化缓存至多四条，新增持久外观向量总上界十条（≤16）；原始 Gallery和状态元数据仍随历史增长，不宣称总内存常数。

组件统计按实际actor代码排除每个组件前16次调用，并非统一按scene时间排除；总Stage2排除前16次native payload调用。Bank promotion在该过滤后没有剩余计时样本，null，不表示没有Bank调用或零开销。完整FPS包含真实图像读取和全部native路径，部署门槛没有通过。

## 最终资格与限制

GO_TRACKING **FAIL**；GO_JEV_INDEPENDENT_VALUE **NOT_ESTABLISHED**；GO_DEPLOYMENT **FAIL**。真实图像诊断的单指标改善不能覆盖原生安全、标签或三种子资格失败，也不能证明动态Question/Option优于普通Fixed/Set。同证据同监督正式比较未获资格，指标null。

核心阻碍是人物历史内容、可区分证据与合法Global ID owner之间的差距。没有历史、只有混合/UNKNOWN owner、真实Bank未提供、视觉证据缺乏区分力的情况无法由增加训练步数解决。下一步应先获得过去可验证的owner区分证据与足够独立reserved纠错事件；若正负外观仍不可分，再独立审计Stage1 ReID。更简单的校准关联/记忆规则应保留为对照，不为坚持JEV包装架构负结果。

## 可复核交付

工程失败和修复保留：P0原始reference路径异常仅CPU修复聚合；P1慢observer保留原始partial并与新轨迹逐条精确核对；P2两次接口合同失败日志保留并修复。没有改变科学门槛或消除原始失败。

代码位于 `gtr/modeling/jev_phase16/` 与 `reproduction_tools/jev_phase16_*`；报告在 `reports/JEV_PHASE16/`，原始大轨迹/日志/权重仅在服务器。各实际阶段绑定actor Git SHA、source/dataset/config/checkpoint SHA、seed、native前缀/RNG和evaluator环境。FINAL_GO_NO_GO.json 区分执行完成与科研失败，SCIENTIFIC_BLOCKERS.json列出后续证据需求。
