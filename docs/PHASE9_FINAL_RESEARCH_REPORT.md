# Phase IX final research report

**本次有条件科研审计已完成：补采的研究回放数量门槛通过，生产 candidate-values 核心通过限定验收；完整数据→生产状态桥接失败，因此在 G_NATIVE 停止，C–F 全部 NOT_RUN。尚不能证明或证伪 JEV 架构的独立收益。**

代码基线 `4190b803ed83bde1a44a015d15d03ceff68e03d0`，分支 `jev/www-jev-phase9-native-candidate-choice-20261008`，worktree `/home/liuyeqiang/WWW_jev_phase9`，独立 runtime `/home/liuyeqiang/WWW_jev_phase9_runtime/20261008_v1`。保护 Phase V–VIII 原 refs/报告与 B2 `f2aa3dd2b564d90bfbfb62dc0518b0b2107a931ed7134f8c0d19f5d152b94ed7`；旧 validation FAIL 永久保留。controller-heldout20/21/22 sealed，未启动 Full24，未读取官方TEST，历史checkpoint删除0，新训练模型/权重0。旧 loader 可读取整个TRAIN注释元数据；controller-heldout不保证foundation训练未见这些视频。

| Gate / 工作 | 真实结果 | 对后续的含义 |
|---|---|---|
| Phase0 | 11仓库固定commit、函数级审阅/许可证，43源码/许可证文件哈希 | 仅设计学习，无外部模型性能主张 |
| A frozen supplement | 48预选、46核验；combined59/61；train74；组17/9 | 数量及研究回放支持PASS，未获完整生产训练资格 |
| B shared production core | 20测试；video12完整1200帧/2399commit compat；8帧existing/NEW mutation | 评分→solver→commit及下一步proposal作用PASS，范围限定 |
| B full causal input/state bridge | 原evidence12不等价；单前缀补首向量PASS；94/221全前缀失败 | BLOCKED_DEPLOYMENT_CONTRACT |
| C tiny / D fair training+ablations | NOT_RUN，metrics=null | 无优化/架构胜负结论 |
| E independent online heldout / F lifecycle | NOT_RUN / SEALED | 无generalization或Unified增量主张 |

**具体失败原因。** production birth 初始化 gallery，回放 memory 从后续WRITE才开始。恢复真实首观测在video16/[17,1]可将evidence12误差从1.0降至2.38e−7；但全前缀审计发现有些bank恢复身份增加hits，却被回放在恢复前筛出的MEMORY写入阶段漏掉。video12/[839,1]身份5恢复；到[851,1]生产gallery1367、回放writes1365。39536个身份前缀中缺1/2/3项为39296/234/6，共94个快照不能用birth-only适配。这个失败限制训练/反事实部署等价，不证明每条历史H32标签错误；普通首项偏移的bank阈值本已有补偿。完整生产state64/归一化/干预后gallery/bank等价没有验收。

单前缀修复结果原有过宽的“training authorized”字段，最终改为false并保存原文件/哈希；数值PASS、最初FAIL、全量FAIL及所有负例均保留。没有为得到正向结果更换验证视频、改Gate、增大网络或重训。

**修改的代码及用途。** 新增三个production模块分开numeric evidence、producer及共享constrained assignment；GTRRCNN显式opt-in集成whole-camera augmented values求解、private NEW dummy、native existing提交、semantic NEW/bank边界和纯观察commit hook；配置默认关闭，候选开启要求JEV enabled/MODEoff，非GMT路径fail fast。原关闭路径保留原SciPy调用。新增冻结分组抽样、事实状态capture、有界H8/H16/H32干预、独立资格finalizer、真实production parity、gallery/state桥接审计和20测试。没有复制外部GPL代码或安装大型外部环境。

候选values比较共享augmented solver；GMT compat仅作为旧rectangular Hungarian→threshold语义锚点，其目标不同，未混作公平架构方法。确定性Torch producer不是学习模型；8帧低于min_track_len50，短段postprocessed输出为空，实际mutation证据是native commits及下一步raw scores。12字段中view_fraction目前为0保留位，不声称已实现该在线信息。

完整video12compatSHA `473d0cc9588f1da0476e7c73134a239805b68407517209ea3ffac2bd315bb4bc`。最新代码短回归existing改变13/15次提交，NEW改变15/15次，后续live score改变14次。没有产生新的learned HOTA/AssA/IDF1/IDSW/MOTA或heldout延迟指标。

**七个研究问题。**

1. **旧JEV为什么被简单规则超过？** Phase VII没有建立state-only MATCH优于同solver规则的证据。缺候选相对信息、纠错支持不足、共享重求解机会是可检验解释；现有证据没有单独识别网络架构因果责任。
2. **补采解决样本不足了吗？** 数量层面是：新46/48、validation59/61、train74，17/9个贡献冲突组；但这不等于59个独立重复，也未解决完整production state资格。
3. **Candidate MLP足够吗？** 未检验，未训练，不能回答足够/不够。
4. **DeepSets/CAMEL context解释全部收益吗？** 未检验。外部实现说明context的合理机制，但不是VisionTrack测得收益。
5. **Question/Action有独立贡献吗？** 未检验；固定WHO的MATCH不能证明question conditioning，必须等数据合格后做同context去Q/A消融。
6. **长期utility比correctness更有价值吗？** 未检验。两类标签已分开、未执行效用为null，仍需native桥接后让所有普通模型享有同监督并做loss消融。
7. **最值得保留的论文创新是什么？** 当前可复核证据是有界冲突组因果采样/审计、共享生产候选值接口及状态表示失败定位。架构、长期决策收益和WWW论文贡献尚未建立，不能包装成已有胜利。

**最小下一步假设（本次未执行）。** 在固定预选事件上直接捕获真正production prefix并执行完整native H32 forks，或证明包含birth和reactivation WRITE的无损adapter；先复核同状态/RNG、state64/evidence12、gallery/bank/hits/计数、current assignment与mutated continuation。保持既有labels/结果/分割，只在新版本追加差异审计，再决定tiny。不能通过扩大模型、开放heldout或Unified绕过契约。

**存储和复核。** 原始证据约7.4 GiB仅保留服务器runtime，Git只存必要源码、compact JSON、配置、SHA和Markdown。dataset manifest含845个新raw artifacts，full-prefix audit另含221个状态哈希；production/feature reports保留各自原始trace路径和binding。早期provenance键/time-vector metadata脚本失败在原运行目录保留；纠正脚本在新版本继续，未覆盖失败。最终可运行verify_jev_phase9_delivery.py只读复核。reproduction_tools/finalize_jev_phase9_research.py记录报告生成逻辑；所有未获资格的计划指标为null。

代码/证据提交顺序：b9e2287外部审计→58be352事前协议→7aa78a8事实混合capture→2534eceproduction core→378d59a/5a487dd执行/严格提交资格→8bd2ca4/fe9e313真实existing继续→82a7391回放数量/完整compat结果→a880e01保留输入FAIL→d8e717f单前缀修复与全量审计→470e68d失败归因及入口防护→后续最终报告提交。审计绑定允许dirty标识并逐文件记录SHA，不能把有dirty的生成报告称作clean-run。

**WHAT DID WE LEARN?** 数据数量、solver正确性、生产提交作用和全量因果状态等价是四个不同条件。本次消除了前两类的部分缺口，也找到了第三类接口作用及第四类失败；在契约失败处停止，使后续实验仍然可解释。
