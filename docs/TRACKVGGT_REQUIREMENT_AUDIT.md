# TrackVGGT 最终需求验收（完成）

本表按原始研究目标、全TRAIN审计补充和DEV12交付补充逐项核对。
教师关系gate为负向，六项完整系统、三组检索配对、两份新strict5和资源收集全部完成。
本验收接受预设范围内负结果及其条件未执行分支，不接受科学/学生成功或普遍无效结论。
原始请求全文保留在对应FINAL_GOAL/URGENT_AUDIT_ADDENDUM/DEV12_NATIVE_ASSET_ADDENDUM中。

| 原始要求 | 权威证据与实际范围 | 当前判定 |
|---|---|---|
| 继承已核验VGGT+GMT、独立分支、不得混入JEV | base861c8710；新分支research/trackvggt-identity-query-www27-20261015；原工作树和独立运行源保留；源码审计及原生权重SHA | 已执行；没有JEV算法/权重输入 |
| 阅读真实VGGT-Det AG/QD、来源与许可证 | SOURCE_AUDIT、phase0_audit、真实official/Det层与attention探针 | 已执行；AG/QD归属上游，未宣称原创 |
| 正确层4/11/17/23、patch_start、RGB/pad、权重兼容 | 实际四层张量、保留位置、attention差异、hook-off/repeat精确检查 | 已执行；FP16默认后端失败证据保留，采用预先限定数值修复 |
| 冻结GMT与VGGT、禁止Stage1/2重训 | 原始411-key checkpoint/config/annotation签名；实际teacher manifests；拟合仅小模块 | 已执行；不是完整backbone微调 |
| 原生1152维、合法候选、严格历史、物理来源键 | 全程GMT审计、V3独立连续采集、完整候选/原生pair/history、时间检查 | 已执行；1152包含原生空间/相机信息，未改称纯1024维身份特征 |
| 困难事件与场景划分、未知/NULL/候选缺失保留 | V2 1000/300；V3 3000 fit且原300不变；纯正例/12个真实NULL/267个masked均保留 | 已执行；训练11场景与验证12分开；未知不强制作为负例 |
| AG-A/B/C、mean/EMA、ROI、DINO/native控制 | 固定V2完整对照和V3匹配扩展；真实150/1000权重、loss、参数量和验证输出 | 已执行；V3不覆盖/删去V2失败结果 |
| AG可学习、软训练与硬选择梯度 | real_backward、teacher_training/crop guards、150软阶段及后续hard relevance训练 | 已执行；没有用无梯度随机采样代替AG |
| K16/24/32 | ag_K_ablation，同一已拟合K24权重上的三种推理K；CPU/GPU离散排名对照 | 已执行限定推理消融；未冒充三组K重新训练 |
| 小目标分桶、full/local/combined、相同DINO裁剪 | crop_inputs_complete、crop_primary_results、真实原坐标patch图、相同234监督查询/256 fit | 已执行224分支；原指令为224或336，不扩大到336挑结果 |
| 五种QD、轨迹条件化与容量控制 | 完整7种VGGT/DINO QD加2132697参数nativeMLP；qd_primary_results | 已执行；last较小，不能单独支持容量归因 |
| 有效训练样本/上游优化预算的真实匹配范围 | fit_population_disclosure核对冻结执行源和实际计数：视觉2201（7NULL）、native/ROI2733（12NULL），QD另复用AG1000 | 已披露差异；DINO视觉顺序精确匹配、crop234严格匹配；不得写成native/ROI有效梯度样本或全部上游预算完全相同 |
| Query shuffle、层权重、正确修复/破坏 | 候选对齐权重L1及原始排列、固定300物理键、known/unknown分开、每个控制的修复交集 | 已执行；whole-history shuffle同时影响AG/decoder，不宣称隔离See-Query因果作用 |
| 合法A(t-1)-B(t)-A(t)、可见mask和NULL | real_cycle_features、真实原生矩阵和非零梯度；128条锁定链、63条可见 | 已执行有界原生子集；B<=4、A当前单检测+NULL，未冒充全场景传输矩阵 |
| lambda0/.1等初始化/更新/RNG对照 | 两种backbone各两次200-update续训；实际batch/cycle哈希、初始loss和终态RNG精确相同 | 已执行；.2无有效身份收益支持而未执行，不任意增权 |
| 教师独立关系gate，不能以小幅scalar改名成功 | teacher_gate核对全部56实际命令/产物/验证标签；ROI/native/DINO/shuffle与known fixes逐项结果 | NO_UNIQUE_VGGT_TEACHER_EVIDENCE；不等于证明所有VGGT表示无用 |
| IRGD及GT-only/KD-only/GT+KD/DINO-KD/GMT-teacher学生 | 锁定design中要求valid teacher relations；student_ablation显式计划种子/空实际种子；STUDENT_RESULTS | 条件未满足而未执行；不是学生训练失败或成功 |
| 轻量学生无VGGT、bounded residual、proposal聚合 | TinyRelationStudent/Adapter契约、真实zero-residual40原生字节相等 | 原型/接入契约已验证；没有已拟合部署学生，不能外推学生FPS/收益 |
| 三个训练种子、完整学习方法DEV5和实际solver/final-ID纠错 | 仅对准入学生适用；原教师无准入；identity_correction_audit明确NULL而非伪造0修复 | 条件未执行；继承的原生三个推理种子不冒充三个训练种子 |
| 完整strict DEV5原生/Native Prior参考 | 六份原生三推理种子完整结果及输入/预测/GT SHA复核 | 已复核继承结果；不是新TrackVGGT收益 |
| 新完整2/4/10流系统、检索、吞吐量与延迟 | 真实OFF/Prior40门禁已PASS；六次完整重放及离线评价、10流strict5 TrackEval注册队列 | 六项全部完成；完整输出/检索、真实命令与输入签名均验证 |
| OFF/Prior完整检测成员与检索请求配对 | system_query_pairs复用原始离线评价器；核对完整框/分数/顺序、请求/GT、共同eligible、得失与修复/破坏 | 2/4/10流全部配对通过，6次全部请求保留；eligible成员相同；修复/破坏分别0/0、1/1、1/1 |
| 新完整HOTA/DetA/AssA/IDF1/CVIDF1/CVMA/MOTA/IDSW/Frag | 两个10流完整strict5 TrackEval输出与raw CLEAR Frag | 新跑两方法均完成，HOTA80.748465/80.840633、IDSW482/351、Frag211/212；与各自参考一致 |
| 完整进程成本、资源、存储边界 | queue monotonic wall、setup/export/loop/component、CUDA峰值、RSS和独立被动采样、NPZ预算 | 全部143阶段登记完成；系统8生产进程1.569694054 GPU-wall-h；队列总10.481369922，另有明确排除项，非历史总账 |
| WWW相关实验与准确系统范围 | 原生跨相机clip检索；本地独立双摄像头视频、GT仅事后评价、所有请求保留 | 完整系统及所有请求的真实检索评价完成；不声称互联网部署、10相机同场景、恒定活跃流、固定工作负载的纯缩放曲线或方法新颖性 |
| 全24 TRAIN分布/GT质量，不重复已有唯一性审计 | all24_dataset_census、gt_quality_summary；复用48原GT SHA与唯一性诊断 | 已完成；不修GT，不把未测漏标写0 |
| 全12关联TRAIN连续因果错误审计 | full_train_completion_audit、原始decisions/rankings/events/detector provenance；169468次主查询 | 已完成；1553 known wrong/missed，集合交叠/候选与排序/最终输出分开 |
| Top4/8/16、正确候选可达、独立人物/时段、是否扩V3 | candidate_headroom、event_type_distribution、scene_difficulty、fit expansion decision、V2/V3表 | 已完成；扩展独立同结构/步数/验证比较；不以查询数充当独立错误数 |
| DEV5历史微小提升解释 | dev5_retrospective_analysis、原生/旧VGGT/Native Prior逐场景输出比较 | 已完成限定回顾；仅scene1有实际solver轨迹，其余候选字段NOT_MEASURED |
| DEV12全1200组GT无关范围预先冻结 | 发布scope45824abd、独立f97cf27f采集、40组OFF/on真实状态/输出门禁 | 已完成；不是只抽26个错误、不是从错误帧拼历史 |
| DEV12完整特征/历史/联合矩阵/Bank/提交与原生等价 | 14557查询、14564观测/提交、2477完整调用、16 descriptor版本；全预测和原决策精确相同 | 已完成；FP32无损；40组有内部状态指纹，旧全程没有内部指纹，不扩大相等范围 |
| DEV12标签分离、26错误覆盖与独立只读消费者 | offline labels、coverage、sealed reader；26/26完整，23 full-pure、21 Top8、2 outside、3 no-pure | 已完成；GT不构建输入，14/26来自同一人，未知状态不强行纠错 |
| DEV12封存SHA/资源/本地交付/GitHub | SEALED_ASSET_READY；seal索引ef7f65d0...；.340162307GPU-wall-h；281762969逻辑字节 | 已完成独立工程交付；不训练CV-PIL、不改别分支权重、不用于VGGT调参 |
| 阶段提交推送、实际命令/数据/权重/环境 | experiment_manifest区分继承与TrackVGGT；raw manifests、teacher/cost CSV、研究日志、阶段commits | 阶段源码/实际结果及日志均保留；最终修订提交推送回执作为最后发布证据 |
| 所有命名交付文件及最终状态 | 原目标9份docs/6份results、审计补充13项、DEV12源码/manifest/schema/交接与SHA | 全部命名文件存在并绑定SHA；内容/原始来源/条件分支/限制另行验收，不以存在性替代完成 |

## 不可外推的证据与记录缺口

原始300验证没有已认证原生实际主求解错误，虽然单行Top1有18个错误。
教师排名修复绝不作为Hungarian/Bank/final-ID修复；这不是一个可证明最终身份纠错率的验证集。
后续完整TRAIN12开发资产覆盖26个实际错误，但不得在已看过标签后加入VGGT模型选择来补造独立验证。

原始训练日志保存了loss/梯度，150/1000端点能做真实fit/validation排名诊断；
没有逐步保存训练Recall/margin曲线，也没有直接逐步crop batch键。
这些字段标为NOT_CAPTURED/DISCLOSED_LIMITATION，不能标成已测，也不从端点伪造曲线。
它们限制报告完整性，不产生一个未执行的新科学假设或可声称的正向结果。

主教师只有一个训练种子、一个held-out关联场景；没有多种子/多场景正向复现。
严格DEV5被原GMT预训练见过，历史TEST接触已披露，本次没有新增TEST调参。
源码/输入的4006个直接SHA复查通过，不等于递归验证全部文件或新做了一次跟踪。

六个新系统用例、两份新strict5指标、资源汇总和最终内容检查均已落地。
最后执行封闭报告索引、当前文件SHA审查及最终提交推送核验；外部目标状态仅在
发布回执和人工范围复核完成后关闭。历史缺失记录仍保留为限制，不改写为已测。

## 最终人工范围审查

已逐项复核原始请求、注册计划、实际命令/源版本、原始产物与SHA、单位测试的
适用范围、数值/输入/因果约束、36个命名文件及当前内容。机器审查的30行中，
27行为限定范围已验证、1行为预设关系gate失败导致条件未执行、1行为历史未保存
监控的披露限制、1行只证明命名文件存在/SHA而不替代内容审查；没有待执行的
独立注册实验或未完成系统用例。实际原始结果再次核对了6个全程用例、2份新strict5、
3组检索配对、6次全部检测请求保留、56个教师报告、15个未训练计划行、
所有终态队列、DEV12信任锚/只读权限和原工作树。两份新MOT的9项核心指标
与各自继承种子14参考精确一致。没有用PASS状态名或文件存在性独自证明完成。

历史监控缺口仍为未取得，不改写为已测。审计补充明确要求“不要重新启动已经完成的合法计算”
（TRACKVGGT_URGENT_AUDIT_ADDENDUM.md）；因此没有为补曲线而重复已完成拟合，
仅用保留检查点做零更新诊断。该限制、非完全相同的native/ROI有效梯度样本/上游预算、
单训练种子/单关联验证场景，以及原300中没有已认证实际主求解错误，均保留并限制结论。
这支持的是已执行预设范围的负结果完成，不支持普遍特征无效、教师/学生成功或PAPER_READY。

最后的发布证据为本分支最终提交、推送回执及远程HEAD核验；只有这些落地后才关闭外部目标状态。
