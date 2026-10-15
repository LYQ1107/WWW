# V2 预注册：监督优先，按证据停止

冻结于新增GPU作业前。授权来源 USER_TASK_SPEC.md；新轮独立最多4 GPU-wall-hours、128MiB持久留存、目标盘余量≥30GiB。旧11.929048h与全部结果保持原账，旧九次计时提案不执行。优先空闲GPU7；不停止其他作业。

P0在TRAIN12/13/14完整原GMT冷启动轨迹上，仅观察冻结MLP step300的反事实分数。使用同一24维pre-commit输入、同一候选与支持长度、相同SciPy rectangular Hungarian和严格原生>0.1×support阈值。原生candidate policy关闭；无额外global re-solve策略。完整计划按整体二选一，不拼接行。实际GMT接受提交逐条核对，并将每段完整raw记录与旧冻结原GMT逐项比较。诊断额外求解与GT标注不算部署FPS。

Q1严格使用已提交的过去证据：known覆盖不足80%或不足3次、未解决别名等为UNKNOWN；两个人各≥3次有效过去观测且known覆盖≥80%为CONFLICTED；first-past owner仍≥80%且唯一别名、无第二个≥3票混杂为RELIABLE。GT只在离线labeler中。Q1诊断输入仅24维中的历史通道6/7/8/9/10/15/18/20/23，其余置零，禁止当前人物GT或未来标签。当前camera提交在观察回调返回后才进入离线历史；跨相机同一场景较早已提交观测属于因果可用历史。

Q2只对两个实际不同的完整accepted计划判断：所有改变行必须唯一目标、两方案均接受已认证历史身份；一方所有改变行正确且另一方至少一个已认证错误才分别KEEP/APPLY。任一改变行拒绝/出生/Bank反事实、不确定目标/owner/别名，就整计划UNKNOWN；相同计划不造选择标签，双方都错/混合权衡不造最优标签。UNKNOWN绝不等于KEEP正确。

Q2训练最小工程准入（非科学充分性）：每类≥8个认证计划、≥8个人物连接依赖组，覆盖≥2个TRAIN视频。共享人物的多个计划即使跨相机/时间块也并为依赖组；另外完整报告独立video-person数与时间块数。时间留出仍为(frame//64)%5==4，该桶矩阵不训练。历史认证跨桶连续，早期留出GT可能影响后来TRAIN标签；不冒充严格标签隔离或独立验证。

全部TRAIN审计后若准入不足，仅重审已有原生日志一次；Rule事件/ORACLE见证不能改名为MLP纠错。随后只做一次Q1可靠性诊断：899参数24→32→3；TRAIN12/13非留出桶训练，TRAIN14仅作Q1诊断。只在训练RELIABLE/CONFLICTED各≥4个原生ID组时训练；UNKNOWN是过去证书不充分类别，不产生Q2动作标签。按video/nativeID/camera/64帧块/标签去重，三类各最多2000个bottom-SHA样本，RAM与分组元数据总计≤32MiB，无持久样本库。

Q1诊断AdamW lr=.001 wd=.0001，FP32，clip1，seed20261015，固定300步，batch≤64，CE按固定训练频数倒数加权；不看TRAIN16选step。每25步记Q1 loss与梯度，Q2明确无标签/无更新。保存固定300步。报告视频14混淆矩阵、置信≥.9的CONFLICTED精度/原生ID组数，并比较固定appearance-consistency<.5且异相机存在的代理规则。该诊断不参与native commit、不证明两问题结构或在线保护收益；新分类器未训练14不等于旧GMT/MLP没见过14。

只有Q2通过准入才实现并训练完整typed/ordinary verifier：冻结GMT和原MLP，≤8000新增参数、32维共享状态、相同输入/监督/样本顺序/300步终点，普通对照容量差≤10%。L=decision CE+reliability CE+10*preserve+.001*regular；各头非零真实监督梯度先验核对。不根据TRAIN16调权或选模型。OFF及零增量默认严格保持MLP；强制KEEP/APPLY选择完整合法分配后单次native commit，先证明Bank/出生和异常恢复契约，再做整视频，不能手工改ID模拟。

开发路径A：HOTA≥冻结MLP+.20，AssA/CVIDF1不低于MLP，IDSW/Frag不增加。路径B：HOTA≥MLP-.05，AssA/CVIDF1各最多下降.05点，IDSW≤.95×MLP，Frag不增，历史锚和回顾性不一致持续量均改善。两路径HOTA均须高于匹配GMT；MOTA等全部指标照实列出。TRAIN16单种子不算独立成功。新模型只有真实在线正向后才做最小Q1/类型/跨相机消融和同卡完整图像FPS；95%门槛使用当轮重新核验GMT。质量不通过不靠参数小或局部标签成绩宣布突破。

P0或Q1诊断没有有效信号即结束本轮结构探索，保留GMT+MLP，记录JEV_VERIFIER_INDEPENDENT_VALUE_NOT_ESTABLISHED。预算上限不是必须用完。科学成功需原目标全部条件与独立视频、训练种子；不新增DEV17–19、封存20–22、TEST、Full24或下载。
