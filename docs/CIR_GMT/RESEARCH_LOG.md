# CIR-GMT 研究日志

2026-10-16：读取完整新任务，创建新Goal，fetch远端确认f595189e；新隔离分支research/cir-gmt-fulltrain-test-20261016。先完成真实调用图审计，再抽取五段正常匹配流程；默认OFF、KEEP只返回原Tensor，初始化不干预。CPU13项真实Instances/Bank/阈值/共享存储回滚合同通过。

TRAIN24/TEST路径清点完成：57508/58038图像无缺失。TEST仅元数据、存在性和SHA，未进行本轮身份评价。所有旧研究成绩不作为新成绩。

Phase0真实前缀v1观察器合同失败：从Git AST读取原函数时拷贝了全局namespace，导致原函数调用绕过后来安装的solver observer；记录失败原日志，不是轨迹等价通过。修正为原函数code绑定真实module globals，重新执行资格检查。未据此改变任何分数、阈值、数据范围或评价标准。

真实前缀v2在被丢弃的分支暂停观察器时缺少被动source字段，Instances.cat合同拒绝。保留日志，丢弃分支补相同来源字段后v3通过：40组真实图像，原函数/OFF/KEEP的全部输出、完整native状态及RNG相等，实际执行并恢复了一个原生图像步骤。v3运行期间新增完整求解矩阵SHA的后续代码未作用于该已启动进程，因此v3不声称已有矩阵SHA核验；后续正式检查使用冻结后新进程。

41项CPU组件测试通过，包含C1公式/1152/严格历史/合法候选和GT接口拒绝、元数据Ledger、0/2/4输出缓冲及当前图像加载器。Ledger/缓冲/C1尚未接入运行中的tracker；这些测试不是原生SPLIT/RELINK资格。未修改的TrackEval和官方MATLAB/MEX通过一次合成完整解析检查，100分只为人工完美输入，不是科学成绩。原native缓存只读SHA核验完成12视频（原11+12），另外12需有界连续扫描；本轮训练视频数仍为0。完整OFF/KEEP在执行，未启动C1或训练。

Phase0完整TRAIN04（901同步组/1802实际图像）完成：Original源码/OFF/KEEP的全部预测、901组native状态、RNG、完整联合cost SHA及solver/阈值/提交记录均一致；OFF真实分支恢复后所有未来保持一致。固定Stage2 strict加载成功。该审计过程保守记账约0.371 GPU-wall-hour；全Phase0累计约0.411，旧账本不变。P0_NATIVE_QUALIFICATION.json通过，科研Goal仍ACTIVE、真实跟踪指标仍null。下一步用实际官方forward的当前组图像适配器完成全视频输出核验，再开放C1；其余41组件测试不等于SPLIT/RELINK准入。

Phase1：Original官方forward在TRAIN04完整901组完成，raw3635条与完整Causal参照精确相同，真实图像7.78655FPS（含read/map/感知/关联/原生状态/后处理，未含模型初始化与评测）。C1接入前后43项CPU合同通过；一条新测试最初创建了长度不一致的fixture，日志保留，改正fixture后通过。新核心OFF/KEEP40组再验通过；C1官方/逐组40组真实图像通过独立CPU公式与全部实际求解/输出/RNG核验，无缺失来源回退。新核心完整OFF已启动。C1仍未有完整视频质量结果。预检从记录SHA的工作文件启动，随后commit封存确切版本；不将当时HEAD字段误写成全部执行源码，后续批量任务用干净固定source worktree。

新核心OFF完整TRAIN04已完成，raw/stock与Original全视频逐项一致。第一次汇总门控误把仅Causal入口使用的online_runner.py当成官方actor报告必含源码，KeyError原日志保留；改为分别验证官方实际导入核心与两个前缀的Causal源码，未改变轨迹、阈值、成绩，v2通过。实现test_net.py显式--cir-run-spec输出驱动并校验DetectionCheckpointer实际加载的所有张量；默认入口不变。数据注册补传原JSON中的frame_id；未新增/推测时间。共享执行函数重构后CPU43项重跑通过，正式入口的完整OFF检查即将运行。
