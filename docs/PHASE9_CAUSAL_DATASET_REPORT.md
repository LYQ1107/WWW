# Phase IX causal dataset audit

**结论：补充样本的回放数量门槛已达成，完整生产状态资格未通过；正式训练未执行。**

基础 Phase VIII `4190b803ed83bde1a44a015d15d03ceff68e03d0` 保持原 TRAIN 122211 / validation 38887 MATCH events、可行候选 2278/118、原因果事件 82/13、纠错74/13和 validation FAIL。Phase IX 使用独立 worktree/runtime，未追认旧 Gate。

冻结协议 SHA256 `5d8504af27a351346e683aeeec1bb67678a71249b213a181bd6bb020d788a966`。TRAIN12/13/14/16，validation17/18/19，controller-heldout20/21/22 sealed。补充48来自既有118个可行索引，先排除原13，再按完整候选引用、历史锚定目标、同一状态和 H32 时间依赖形成传递冲突组。种子20261009，视频配额17/14/17、组内无放回抽样；未来 utility 之前已提交冻结方案，失败与 unknown 全保留。

| 回放支持 | 审计事件 | 核验纠错 | 贡献视频 | 核验冲突组 |
|---|---:|---:|---|---:|
| 原 train | 82 | 74 | 12/13/14/16 | 17 |
| 原 validation | 13 | 13 | 17/18/19 | 9 |
| 新 supplemental | 48 | 46 | 17/18/19 | 8 |
| combined validation | 61 | 59 | 17/18/19 | 9 |

验证按视频核验24/14/21（原+补充）。新48中2个 video18 事件未满足立即正确提交等完整纠错条件，仍在 audit 中；不能因为全部 CANDIDATE_0 H32 为正就把这两行算成合格纠错。原13、补充48、合计61的组权重 Kish ESS 分别11.5714/35.3135/43.6574；这些数不是独立重复次数，真正组数为9/8/9，且仅三个验证视频。

预注册同时采集 natural56（train32/val24）、hard negative13（10/3）、unknown14（8/6）；加补充48，共131条采样记录/131行、130个唯一状态快照。部分视频达不到每视频4个困难负例，使用全部实际可用事件，未人工创造样本。自然分布的采样概率保留，候选级普通指标/神经模型自然分布性能未计算，不能用纠错富集分布代替总体收益。

有界未来预算执行67个新事件、354个分支；原95保留，compact audit 共162个事件。CONTROL/KEEP_FACTUAL 使用相同完整回放状态和 RNG，各 CONTROL 的实际 ID 与完整事实流一致。干预包括候选、替代、共享求解 REASSOCIATE、semantic NEW；之后重新计算 live GMT proposal。H8/H16/H32、实际提交、行级立即正确性、全场效用及 birth penalty=0 敏感性分别记录。效用继承 Phase VIII 冻结权重：correct − wrong − 5 merge − .25 birth。未执行候选效用始终 null；依赖其他行的完整 assignment 效应不等同穷举 Q 函数。错改、负效用、tie、unknown 未删除；身份错误时长按 camera-frame 记录，未假定 FPS 转秒。

候选 correctness 来自永久前缀 anchor，整数 GT ID 不与 track ID 直接比较；多正确 alias/unknown 保留。在线候选来自实际 GMT，全候选集，无 GT Top-K 保留。GT 仅用于离线审计选择和标签。当前新捕获行已有1635条已知 existing-candidate correctness 标签，但 NEW 正确性没有正例（train44 unknown/6 false，val70 unknown/11 false），不能伪造 NEW 监督，也不能把未训练称为优化失败。

**后续完整桥接审计改变了训练资格。** 直接将 replay.memory 当生产 gallery，train16/[17,1] 的 evidence12 最大偏差1.0。恢复真实首观测后该单前缀误差降至2.38e−7，但221个全前缀中94个不满足仅补首向量的契约。共39536个身份前缀，hits−writes=1/2/3分别39296/234/6。生产 birth 初始化 gallery；回放 memory 不含该项；bank 恢复后生产还写入观测，回放先筛 final_existing 的 MEMORY 阶段会漏掉恢复行。video12/[839,1] 身份5恢复，在[851,1]生产 gallery1367条、回放1365条，证据见 GALLERY_FAILURE_ATTRIBUTION.json。

这证实输入/状态映射尚未统一，不证明所有已有 H32 数值都错。普通首项偏移的 bank eligibility 阈值已有补偿，也不能直接声称所有 bank 支持都不同。完整 native 干预后 bank/gallery/normalization/state64 等价尚未获证，因此数字 PASS 仅限研究回放支持，G_NATIVE 为 BLOCKED_DEPLOYMENT_CONTRACT。

原始快照、效果和轨迹保留 `/home/liuyeqiang/WWW_jev_phase9_runtime/20261008_v1`；845个新 dataset artifacts（130 state、354 EFFECTS、354 TRACE、7 fork manifest）的路径/SHA见 PHASE9_CAUSAL_DATASET_MANIFEST.json。原状态来源及哈希还见 MEMORY_REPRESENTATION_AUDIT.json。Git仅收 compact 标签、指标、清单、代码与文档。早期脚本 provenance/time metadata 错误运行亦保留，未作为科学失败样本删除。

**WHAT DID WE LEARN?** 事前分组补采能解决观察到的数量不足，但回放自洽和实际 production ID 一致都不能替代完整状态表示与线上输入的等价证明。
