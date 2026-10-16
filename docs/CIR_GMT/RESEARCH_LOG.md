# CIR-GMT 研究日志

2026-10-16：读取完整新任务，创建新Goal，fetch远端确认f595189e；新隔离分支research/cir-gmt-fulltrain-test-20261016。先完成真实调用图审计，再抽取五段正常匹配流程；默认OFF、KEEP只返回原Tensor，初始化不干预。CPU13项真实Instances/Bank/阈值/共享存储回滚合同通过。

TRAIN24/TEST路径清点完成：57508/58038图像无缺失。TEST仅元数据、存在性和SHA，未进行本轮身份评价。所有旧研究成绩不作为新成绩。

Phase0真实前缀v1观察器合同失败：从Git AST读取原函数时拷贝了全局namespace，导致原函数调用绕过后来安装的solver observer；记录失败原日志，不是轨迹等价通过。修正为原函数code绑定真实module globals，重新执行资格检查。未据此改变任何分数、阈值、数据范围或评价标准。

真实前缀v2在被丢弃的分支暂停观察器时缺少被动source字段，Instances.cat合同拒绝。保留日志，丢弃分支补相同来源字段后v3通过：40组真实图像，原函数/OFF/KEEP的全部输出、完整native状态及RNG相等，实际执行并恢复了一个原生图像步骤。v3运行期间新增完整求解矩阵SHA的后续代码未作用于该已启动进程，因此v3不声称已有矩阵SHA核验；后续正式检查使用冻结后新进程。

41项CPU组件测试通过，包含C1公式/1152/严格历史/合法候选和GT接口拒绝、元数据Ledger、0/2/4输出缓冲及当前图像加载器。Ledger/缓冲/C1尚未接入运行中的tracker；这些测试不是原生SPLIT/RELINK资格。未修改的TrackEval和官方MATLAB/MEX通过一次合成完整解析检查，100分只为人工完美输入，不是科学成绩。原native缓存只读SHA核验完成12视频（原11+12），另外12需有界连续扫描；本轮训练视频数仍为0。完整OFF/KEEP在执行，未启动C1或训练。

Phase0完整TRAIN04（901同步组/1802实际图像）完成：Original源码/OFF/KEEP的全部预测、901组native状态、RNG、完整联合cost SHA及solver/阈值/提交记录均一致；OFF真实分支恢复后所有未来保持一致。固定Stage2 strict加载成功。该审计过程保守记账约0.371 GPU-wall-hour；全Phase0累计约0.411，旧账本不变。P0_NATIVE_QUALIFICATION.json通过，科研Goal仍ACTIVE、真实跟踪指标仍null。下一步用实际官方forward的当前组图像适配器完成全视频输出核验，再开放C1；其余41组件测试不等于SPLIT/RELINK准入。

Phase1：Original官方forward在TRAIN04完整901组完成，raw3635条与完整Causal参照精确相同，真实图像7.78655FPS（含read/map/感知/关联/原生状态/后处理，未含模型初始化与评测）。C1接入前后43项CPU合同通过；一条新测试最初创建了长度不一致的fixture，日志保留，改正fixture后通过。新核心OFF/KEEP40组再验通过；C1官方/逐组40组真实图像通过独立CPU公式与全部实际求解/输出/RNG核验，无缺失来源回退。新核心完整OFF已启动。C1仍未有完整视频质量结果。预检从记录SHA的工作文件启动，随后commit封存确切版本；不将当时HEAD字段误写成全部执行源码，后续批量任务用干净固定source worktree。

新核心OFF完整TRAIN04已完成，raw/stock与Original全视频逐项一致。第一次汇总门控误把仅Causal入口使用的online_runner.py当成官方actor报告必含源码，KeyError原日志保留；改为分别验证官方实际导入核心与两个前缀的Causal源码，未改变轨迹、阈值、成绩，v2通过。实现test_net.py显式--cir-run-spec输出驱动并校验DetectionCheckpointer实际加载的所有张量；默认入口不变。数据注册补传原JSON中的frame_id；未新增/推测时间。共享执行函数重构后CPU43项重跑通过，正式入口的完整OFF检查即将运行。

正式test_net显式入口完整TRAIN04 OFF通过，raw/stock、全部图像SHA和归一化冻结GMT配置与Original精确一致；实际加载权重逐张量核验通过。46项CPU测试通过（新增输出目录/方法差异与GMT阈值不可更改的配置合同）。将从干净固定source worktree运行完整C1。

完整C1已在干净3e6d6c5d source经test_net完成TRAIN04：原生主关联1800调用，1771次实际全矩阵重求解，164行相对当时原生提案改变，2个原先接受行被改变；这些不是164次认证纠错。无来源fallback、VGGT/DINO调用和新增参数。raw HOTA74.736869→75.071076、AssA71.504150→72.158622、官方CVIDF188.508361→89.016722、IDSW355→296；DetA78.200580→78.183045、MOTA81.458333→82.994792、Frag40→40，完整图像FPS7.786552→7.720871。stock HOTA75.575567→75.844408、CVIDF192.505593→92.738676，IDSW4→4、Frag71→68。仅FIT工程场景，不据此选择超参数。离线固定过去owner审计：2条WRONG→CORRECT均在1个人物组，0条CORRECT→WRONG，21条CORRECT→UNANCHORED单列；UNKNOWN不算修复。

51项CPU合同通过。已按原冻结12个完整DEV视频启动Original与C1共24任务，使用空闲GPU4–7、同一个干净3e6d6c5d source。按7.72图像FPS估计约2.21GPUh，单独注册5GPUh保守阶段上限（CIR累计授权6，不重置任何旧账本）；预算/进程/每视频日志持续记录。所有12均运行；7个原始GT存在重复事件的视频保留完整预测，严格官方metric失败应null，不改GT或评测合法性检查。TT和最终24拟合仍未运行。

P1全12视频Original/C1共24个真实完整任务已完成，全部从同一干净3e6d6c5d源运行。合法5视频原始GT指标先完成：raw HOTA80.748757→80.822127、AssA81.472600→81.625552、CVIDF195.057962→95.251576、IDSW482→371；stock HOTA81.055072→81.036274、CVIDF195.631664→95.661845、IDSW41→48。raw小幅正向、stock存在退化，不宣称全面提升或最终Goal成功。原GT有冲突的其余7视频保留完整预测，指标null。原始11视频中间评价完整保留，最后23视频也完成后验证其原始GT无效，再复用完全相同合法5视频的native组合计数，不重复算作新实验。

按照P2已冻结条件，增加唯一bias+2普通接受阈值解释控制，复用原生完整Hungarian/Bank/提交，不加载身份特征或训练参数。CPU64合同通过；新core还需原始OFF、KEEP及原C1/bias2真实前缀再验。登记至多0.4GPUh，从已注册CIR累计6小时的未用额度内分配；旧JEV/PIRGA账本不改。P2 FIT12规则审计仍CPU运行，先生成完整GT-free输出，再解析监督，尚未训练或启用Guard。
