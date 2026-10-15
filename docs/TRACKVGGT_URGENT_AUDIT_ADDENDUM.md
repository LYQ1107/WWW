# URGENT ADDENDUM — Full VisionTrack Training Audit & Hard Identity Association Mining

## 0. 任务性质与优先级

这是对现有 TrackVGGT-MOT FINAL GOAL 的补充指令。

**不是取消原来的 AG、QD、TCVG、CVTIC、IRGD 三个研究方向，而是在正式扩大教师训练之前，先解决训练数据不足、困难样本覆盖不足，以及评测子集过于容易的问题。**

你必须将本任务增加到当前 FINAL GOAL 中，作为数据与关联错误诊断的高优先级前置阶段。

当前研究分支：

`research/trackvggt-identity-query-www27-20261015`

已检查的最新参考提交：

`0c4a3a94a9a51ea8f6d05b9756b70133182ebfdc`

开始执行前重新检查 HEAD、工作树状态和运行中的任务。

不要覆盖已经推送的代码。

不要删除已经完成的困难事件数据。

不要中断正在正常运行的 VGGT/DINO 特征提取任务。

如果已有训练进程正在使用旧的固定样本，请保留其运行结果；新的全量审计应在独立工作树、独立输出目录中执行，并将后续扩大数据的实验设为独立对照。

**此次目标不是盲目增加训练样本，而是准确回答四个问题：**

1. 现有 GMT 在整个 VisionTrack TRAIN 中究竟还有多少真实身份关联错误？
2. 这些错误中，有多少属于正确候选存在、但 GMT 关联失败的可修正错误？
3. 当前仅使用 1000/300 个查询，是否遗漏了其他场景的大量有效困难身份事件？
4. 如果扩大训练数据，是否能够为 TrackVGGT 的 Identity-Guided AG 和 Track-Conditioned QD 提供更充分的监督？

要求立即执行数据审计，完成后自动形成报告，再决定是否扩大教师训练样本。

---

# 1. 先理解目前已经完成了什么

你必须首先阅读：

`docs/TRACKVGGT_HARD_EVENTS.md`

`docs/TRACKVGGT_EXPERIMENT_PLAN.md`

`docs/TRACKVGGT_RESEARCH_LOG.md`

`manifests/geo_split_v1.json`

`manifests/trackvggt_design_v1.json`

`manifests/trackvggt_fit_expansion_v2.json`

`manifests/trackvggt_phase2_tokens_v2.json`

`results/trackvggt_expanded_cohort_v2.json`

检查已有困难事件文件及 SHA256。

现有实验已经完成：

- TRAIN Scene 4、5、6、7、8 的困难事件收集。
- 固定训练查询总数 1000。
- TRAIN Scene 12 的验证查询 300。
- 训练集合中 130 个可到达正确候选的已知排序错误或主关联失败事件。
- 其中明确的 GMT Top-1 排序错误共 19 个。
- 现有验证集合在 Scene 12 中存在严重的小目标 patch 覆盖问题。

这些数字只对应现有选中查询，不是全部原始 TRAIN 中的错误数量。

因此不允许直接推断整个 VisionTrack 很容易。

也不允许把已经完成的 1000/300 样本称为从未开展困难样本挖掘。

**新增研究必须在现有工作基础上继续。**

---

# 2. 审计范围：全部原始 TRAIN 的 24 条序列

读取 `manifests/geo_split_v1.json`，严格使用已有划分。

现有划分：

## 2.1 关联训练范围：12 条

视频 ID：

`[4,5,6,7,8,9,12,13,17,20,21,24]`

需要完成全部序列覆盖审计。

其中：

- 4、5、6、7、8 已有 400 组同步帧的困难事件收集。
- 12 为现有的关联训练内部验证序列。
- 9、13、17、20、21、24 尚需重点检查实际困难事件。
- 已扫描过 400 组的序列，也不代表完成整段视频扫描。

不要把 400 组样本当成全视频。

## 2.2 严格 DEV：5 条

`[1,10,11,18,19]`

已经有完整因果评测结果。

本轮优先复用现有预测和评测报告。

不要为了统计身份错误重新运行大量 GMT 或 VGGT。

这些数据只能用于事先固定的开发评价或事后机制诊断。

不能根据这 5 个场景的 GT 错误直接改变训练样本选择和模型结构。

## 2.3 其他 DEV：7 条

`[2,3,14,15,16,22,23]`

这些序列已有原始 GT 身份冲突记录。

只做标注质量统计和明确范围的诊断。

不允许删除冲突行后，将它们伪装成官方严格评测数据。

如果能够明确界定无冲突观测，只能作为单独标注的诊断结果，不得并入原有严格 DEV5 指标。

## 2.4 官方 TEST

本次不允许使用官方 TEST 的 GT：

- 不用于困难事件挖掘。
- 不用于训练样本选择。
- 不用于超参数调整。
- 不用于决定 AG/QD 结构。
- 不用于教师与学生模型选择。

此次是 TRAIN 内部的数据与错误审计，不是重新定义正式测试协议。

---

# 3. Phase A：先审计 24 条序列的基本分布

本阶段优先使用 CPU 和已有标注文件。

先读取已有：

`results/train_gt_quality_audit.json`

以及相关标注统计。

不要重复已经完成且哈希一致的完整 GT 唯一性审计。

只补充缺少的字段。

每条序列统计：

- 视频 ID。
- 场景名称。
- 摄像头数量。
- 每个摄像头帧数。
- 同步帧组数。
- GT 目标实例数。
- 去重后的真实身份数。
- 同时出现的目标数量分布。
- 跨摄像头同时出现的真实身份数。
- 每个身份的轨迹持续长度。
- 单摄像头可见与多摄像头可见身份比例。
- GT 框像素尺寸分布。
- 与 518×518 VGGT 输入对应的 patch 覆盖分布。
- GT 重复、冲突、缺失或无效记录数量。

目标尺寸至少分为：

- 无有效 patch 中心。
- 1–3 个 patch。
- 4–15 个 patch。
- 至少 16 个 patch。

注意 patch 覆盖不等同于目标真实信息量。ROIAlign 可能仍能从邻近特征中提取信息。

这一阶段的目的是判断：

**不同场景是否具有不同的多摄像头身份关联难度，以及 VGGT 的默认分辨率是否对部分场景明显不足。**

输出：

`results/trackvggt_all24_dataset_census_v1.csv`

`results/trackvggt_gt_quality_summary_v1.json`

`docs/TRACKVGGT_DATASET_DIFFICULTY_AUDIT.md`

要求覆盖全部 24 条原始 TRAIN 视频。

该阶段不需要调用 VGGT。

---

# 4. Phase B：完整 GMT 因果跟踪错误审计

这是最重要的部分。

必须使用真实 GMT 跟踪状态，而不是使用单帧 GT 构建理想身份历史。

## 4.1 复用已有代码

首先阅读：

`geo_tools/collect_track_hard_events.py`

`geo_tools/audit_track_events.py`

`geo_tools/prepare_trackvggt_cohort.py`

`geo_tools/collect_hard_events.py`

`geo_tools/audit_tracking_anchor_errors.py`

`gtr/geometry/decision_trace.py`

`gtr/geometry/online_runner.py`

其中 `collect_track_hard_events.py` 已经具备：

- 实际 GMT 关联信息记录。
- 真实历史 track_id。
- Candidate logits。
- Proposal-to-track 分组。
- 主关联决策记录。
- 原始 GT 后置标注。
- OFF40 prediction parity。
- 已知正确、错误、模糊和未知状态区分。

**必须在现有工具基础上扩展，而不是新写一套完全不同的跟踪器。**

## 4.2 全视频扫描方式

当前 collector 默认：

`--max-frames 400`

`--limit 500`

这只是有限采样，不能代表完整序列。

新增明确的全视频扫描模式，例如：

`--full-video`

全视频模式必须遍历真实场景的全部同步帧组。

不能在没有检查 `video_records()` 和 collector 参数语义时，随意传入 0 或超大数字假装全量。

建议新增一个只读流式审计入口：

`geo_tools/audit_full_train_identity_events.py`

该工具需要支持：

- `--video-id`
- `--full-video`
- `--output`
- `--seed`
- `--summary-only`
- `--max-retained-events`

当 `--summary-only` 启用时：

仍然统计全部关联决策。

只对少量需要诊断的事件保存详细状态。

避免将全部历史 GMT 向量和候选矩阵长期保存到磁盘。

### 重要

完整审计必须从视频第一帧连续跟踪到最后一帧。

不能每隔 20 帧重新初始化跟踪器。

不能用 GT 重建历史轨迹。

不能为了便于标注而替换原始模型预测。

必须保持原 GMT 的真实在线关联行为。

---

# 5. Phase C：明确划分 GMT 错误类型

禁止再将“不知道是否正确”当作“匹配错误”。

对于每次实际 GMT association query，必须标注：

## Type A：正确候选存在，排序错误

当前检测有明确 GT 身份。

完整历史候选中存在身份纯净且 GT 相同的合法轨迹。

GMT 排名最高的是另一个身份已确认的轨迹。

这是 AG/QD 最有希望解决的错误。

## Type B：正确候选存在，但关联拒绝或分配失败

候选中存在正确的历史轨迹。

但实际 Hungarian 和阈值处理没有分配正确 ID。

这里要再区分：

- 匹配分数不足。
- Hungarian 竞争冲突。
- 无匹配概率导致拒绝。
- 轨迹长度阈值影响。

这一类型有可能通过改进关系分数解决，但不能默认全部都是视觉特征错误。

## Type C：正确轨迹存在于完整历史，但被 Top-8 删除

必须统计：

- Top-4 覆盖。
- Top-8 覆盖。
- Top-16 覆盖。
- 完整候选集合覆盖。

如果正确身份在完整历史中存在，却被 Top-K 过滤掉，不能把它当作普通的 unmatched。

本次需要明确区分候选召回问题与候选排序问题。

## Type D：正确身份没有出现在合法历史候选中

可能来自：

- 漏检。
- 新目标。
- 历史轨迹丢失。
- 当前合法历史窗口无法覆盖。
- 已有历史身份污染。

这类事件通常不能仅通过对现有候选重排序解决。

必须单独报告。

## Type E：历史轨迹身份混乱

一个预测 track_id 对应多个 GT 身份，或其参考观测中存在无法可靠赋予 GT 的对象。

这属于 mixed/impure/unknown reference。

不能随便给整个历史轨迹指定一个 GT ID。

特别注意：

正确 GT 观测存在于混合历史中，不等于存在一个可直接使用的纯净正确轨迹。

## Type F：无法可靠判断

包括：

- 当前检测没有可靠 GT 对应。
- 候选 GT 冲突。
- 参考轨迹身份模糊。
- GT 时间或摄像头对齐不能验证。

这些事件可用于无监督诊断，但不能作为明确的监督正负样本。

---

# 6. Phase D：完整统计可修正错误空间

每条完整训练视频必须输出以下计数。

N_total：

全部实际 GMT 主关联 query 数。

N_known：

能够可靠获得当前查询 GT 身份的 query 数。

N_wrong：

已知身份错误的 GMT 决策数。

N_rank_correctable：

正确纯净身份候选存在，但 GMT 排序错误的事件数。

N_assignment_correctable：

正确候选存在，但 Hungarian/阈值造成错误或拒绝的事件数。

N_missing_candidate：

正确身份在当前合法候选集中不存在的事件数。

N_impure：

历史 reference 身份混乱的事件数。

N_unknown：

无法可靠确认正确性的事件数。

N_top4、N_top8、N_top16：

在合法 GT-known 查询中，正确纯净候选进入对应 Top-K 的次数。

N_cross_camera：

实际需要跨摄像头身份关联的事件数。

N_same_camera：

同一摄像头内的历史关联事件数。

**这些定义的集合可能存在包含或重叠关系，必须在报告中注明。不能将所有计数无条件相加。**

另外分别统计：

- GTA 排序前后的正确率。
- Hungarian 实际选择。
- 阈值接受或拒绝。
- Memory reactivation 后的改变。
- 最终输出 ID 正确性。

不能将 solver 内部纠错次数等同于最终 CVIDF1 改善。

## 6.1 两种不同的潜力指标

定义已知错误中的候选可修正比例：

`CorrectableErrorRate = N_correctable / N_wrong_eligible`

分母仅包含 GT 与候选状态可可靠判定的事件。

另定义正确候选覆盖率：

`CandidateCoverage@K = queries_with_pure_positive_in_topK / known_queries_with_pure_positive_in_full_legal_history`

必须给出分子和分母。

不能只输出百分比。

这些比例用于判断 AG/QD 的研究空间，不能直接称为 HOTA 或 CVIDF1 的理论上界。

## 6.2 单独估计上界

允许在离线 TRAIN 分析中模拟：

“假设在已有合法候选中始终选中正确身份，最多可修正多少已知错误？”

但要明确：

- 这是 GT 辅助离线诊断。
- 不是合法在线模型。
- 不是完整未来轨迹的真实反事实结果。
- 不能据此声称 HOTA 一定提升多少点。

---

# 7. Phase E：优先扫描哪些序列

考虑时间有限，按两层推进。

## 第一层：完成当前关联训练的未覆盖场景

优先新增完整因果审计：

`[9,13,17,20,21,24]`

这些是此前 AG/QD 训练事件尚未充分覆盖的原始关联训练场景。

开始前依据真实场景长度和当前设备资源建立扫描计划。

先执行少量预检，验证输出与原 GMT 完全一致。

然后执行完整视频。

## 第二层：补齐已经扫描过的前 400 组

对：

`[4,5,6,7,8]`

从原有 400 组之后补齐剩余时间段。

如果使用整个序列重跑是保持历史状态所必需的，允许从第一个同步帧开始，但不得覆盖已有原始输出。

同一模型、种子、配置、协议下，先验证前 400 组与已有结果的正确性关系。

任何不一致先诊断随机数、模型版本、场景读取顺序和插桩影响。

不能把两个不同来源的关联历史直接拼接起来。

## 第三层：保留 Scene 12 验证独立性

Scene 12 当前承担训练内部验证角色。

可以完成独立的全视频审计，但不得用它的 GT 错误重新选择训练事件、训练阶段或模型参数。

现有 300 个验证查询保持原样。

完整 Scene 12 诊断需单独存放，并注明是否已被研究决策查看。

如果新的全量训练方案已经在 Scene 12 上做过模型选择，不得再将其称为完全未参与开发的验证结果。

---

# 8. Phase F：检查是否真的需要扩大训练样本

现有固定集合：

- Fit：1000。
- Validation：300。

保持不变，并作为第一组训练配置：

`Cohort-V2`

新增全量扫描后，建立候选池：

`FullTrain-Hard-Pool`

不要立即替换原始 1000 个训练样本。

必须先生成：

`results/trackvggt_full_train_event_inventory_v1.json`

其中列出全部合法事件的统计及索引。

然后决定是否创建：

`Cohort-V3`

## 8.1 扩大条件

优先看：

- 可修正已知错误是否显著多于原有 130 个混合错误/失败事件。
- 是否有更多明确 Top-1 错误，而不是只有模糊历史。
- 错误是否覆盖至少多个不同场景。
- 是否存在足够多正确候选和困难负候选。
- 是否发现之前没有覆盖的视角、尺寸、时间间隔类型。

如果足够，允许将训练规模从 1000 扩大至约 2000–4000 个查询。

具体数量取决于真实合法样本，不得人为重复事件凑到 4000。

## 8.2 采样规则

扩大样本应包括：

A. 可修正的真实 GMT 排序错误。

B. GMT 高歧义但尚未证实错误的合法事件。

C. 中等难度的正常跨摄像头对应。

D. 容易且已经正确的样本作为稳定性对照。

E. Unmatched 与候选缺失事件。

对于 E 类，要使用符合真实语义的标签与 mask，不允许将 impure/unknown 一律当成真正的新身份。

训练可以增加错误样本占比，但不能只训练错误样本。

建议在已有固定划分内先试一个有限的新训练设置，不进行无限随机采样搜索。

使用场景轮转采样，避免 Scene 8 或某一段密集轨迹垄断全部训练事件。

必须统计独立场景数、独立身份数、独立事件数及相邻帧关联程度。

## 8.3 数据存储

只建立轻量事件索引和必要的 GMT 历史摘要。

VGGT 特征不在全量扫描阶段计算。

等确定新 Cohort-V3 后，才对新增加的选中事件执行 VGGT/DINO 对应 token 提取。

旧 Cohort-V2 已经完成的特征如果配置和哈希一致，可以复用。

新增教师 token 仍遵循现有约 2GiB 的受控缓存预算。

超出预算先报告并采用分批读取，而不是直接无限增长。

---

# 9. Phase G：分析为什么原来的严格 DEV5 提升这么小

从已经保存的完整预测文件中做事后分析。

当前 GMT 基线不同场景的 CVIDF1 约为：

Scene 1：94.665。

Scene 10：98.879。

Scene 11：92.012。

Scene 18：95.944。

Scene 19：94.052。

旧 VGGT 增益主要集中在 Scene 1，约 +0.807 个百分点。

需要分析：

1. Scene 1 的 GT 可确认身份错误是否更多？
2. Scene 1 的正确历史候选覆盖是否更好？
3. Scene 1 是否存在更多可修正的跨摄像头错误？
4. 其他场景是否主要受检测或候选缺失影响？
5. 不同场景目标大小是否明显不同？
6. 不同场景中 VGGT 的 token 覆盖率是否不同？
7. 是否有一小部分场景的输出决定了完整 DEV 的大部分提升？
8. Scene 1 的收益是否对训练头和推理随机种子保持稳定？

这一阶段只能做机制分析。

不能根据 DEV5 的 GT 来重新挑选训练事件、层、采样权重或超参数。

如果 Scene 1 的收益来源仍不确定，应明确报告，而不是猜测成因。

---

# 10. Phase H：重新评价 TrackVGGT 的样本充分性

完成全量训练审计后，生成三个关键表。

## 表 A：每场景真实困难程度

列：

Scene、Frames、Targets、Known Queries、Known Wrong、Pure Positive Available、Top8 Coverage、Impure References、Unknown Rate。

## 表 B：不同错误类型分布

列：

Scene、Ranking Error、Hungarian Error、Threshold Rejection、Missing Candidate、Mixed History、Unmatched、Detection Failure。

这里的事件计数要注明互斥性与判定范围。

## 表 C：不同目标尺寸下的研究空间

列：

Patch Coverage、Query Count、Known Error Count、Correctable Rate、GMT Recall1、VGGT ROI Recall1（若已有合法同批结果）、DINO Recall1（若已有）。

没有进行过同批基础模型评价时，该列应留空并标记 NOT_MEASURED。

不允许编造。

---

# 11. 审计完成后如何决定下一步

## 情况 A：找到了大量真实可修正错误

保留已有 TrackVGGT 设计。

使用更充分的 TRAIN 错误集合，重点训练：

- Identity-Guided AG。
- Track-Conditioned QD。
- Temporal Identity Consistency。

优先验证它们是否比普通 VGGT ROI、DINO 和 GMT 更有效。

下一轮教师训练至少比较：

- Cohort-V2：原 1000。
- Cohort-V3：新扩大训练集合。

两组必须使用相同网络结构、训练预算规则和相同固定验证 cohort。

如果扩大数据后结果明显改善，可以进一步补充样本规模曲线。

## 情况 B：真实身份错误很多，但正确候选经常缺失

不能继续假设问题主要是外观特征不足。

生成：

`CANDIDATE_COVERAGE_BOTTLENECK`

分析 GMT 的合法候选构造、历史轨迹覆盖和匹配状态。

可以在原研究的独立问题分析中研究候选覆盖，但不得随意替换三个创新点或修改原 GT。

## 情况 C：可修正错误极少，且 GMT 已经很高

生成：

`LIMITED_CORRECTABLE_ERROR_HEADROOM`

明确说明当前场景难以验证新身份表示的实际收益。

此时不要盲目扩大教师参数或重复训练。

应当评估现有数据协议是否足以支撑论文结论，以及是否需要之后寻找具有合法授权、明确身份 GT 的更困难外部基准。

这不是授权你当前重新下载和构建大型数据集。

本轮先完成已有 VisionTrack 的科学诊断。

## 情况 D：主要问题是小目标无有效 patch

保留 AG/QD。

优先比较：

- 全图 VGGT tokens。
- 检测框周围扩大裁剪。
- 224/336 局部编码。
- 相同成本下的 DINO 局部编码。

不能预先宣称 VGGT 不适合小目标。

必须判断问题是图像分辨率造成的，还是表示本身缺少身份判别力。

---

# 12. 绝对禁止的审计错误

不得：

1. 使用 GT 生成在线历史轨迹。
2. 在主推理时读取 GT 来选择候选。
3. 用未来帧确定当前 identity query。
4. 将 unknown/impure 当成已知错误。
5. 将 correct observation in impure track 当成 pure positive track。
6. 将 Top-8 候选缺失当成真正 unmatched。
7. 将单个长期身份产生的数千个 output pairs 当成数千次独立正确关联。
8. 只统计选中的 1000 个事件，却声称审计过完整 TRAIN。
9. 将 400 个同步帧组称为完整序列。
10. 通过重设随机种子改变 GMT 原始错误分布后不记录。
11. 修改原始 GT 以增加合法训练事件。
12. 把离线全数据扫描的 FPS 作为 TrackVGGT 实时部署速度。
13. 覆盖或删除已经锁定的旧 Cohort-V2。
14. 提前使用完整 DEV5 指标决定训练样本。
15. 使用官方 TEST 选择模型。
16. 为扩大数据规模重复保存所有原始图像或 dense VGGT 特征。
17. 因为某个场景困难事件很少就重复同一物理 query 充数。
18. 在实际结果没有完成时生成虚假统计或 PASS 报告。

---

# 13. GitHub 与执行管理

此次是对现有 FINAL GOAL 的补充。

不能停止整个 TrackVGGT 项目。

需要检查正在执行的模型特征提取和训练任务。

如果正在运行的任务属于之前固定 Cohort-V2，应允许它们正常完成并保存结果。

不要在其运行期间修改它们依赖的源码和输出目录。

全量审计可以创建隔离工作树，并在无冲突时将经过验证的代码、结果和报告合并回研究分支。

运行设备必须从实际 GPU 占用检测。

已有机器为 V100 环境的记录，不能假设现在一定是 A100。

之前分配的设备为 GPU0/4/5/6，但仍然需要检查实际占用。

不得杀死其他研究进程。

每完成一个阶段，保存：

- Git SHA。
- 精确执行命令。
- 数据来源。
- 视频 ID。
- 实际同步帧组数。
- 模型 checkpoint SHA。
- 配置 SHA。
- 预测文件 SHA。
- GT 原始文件 SHA。
- 实际错误计数。
- 运行时间与 GPU。
- 是否完整视频。
- 是否含训练标签辅助。
- 是否通过真实因果及 OFF 等价验证。

然后提交并推送 GitHub。

---

# 14. 最终输出文件

必须生成：

`docs/TRACKVGGT_FULL_TRAIN_AUDIT.md`

`docs/TRACKVGGT_DATASET_DIFFICULTY_AUDIT.md`

`docs/TRACKVGGT_CORRECTABLE_ERROR_ANALYSIS.md`

`docs/TRACKVGGT_FIT_DATA_EXPANSION_DECISION.md`

`results/trackvggt_all24_dataset_census_v1.csv`

`results/trackvggt_full_train_error_census_v1.csv`

`results/trackvggt_candidate_headroom_v1.json`

`results/trackvggt_event_type_distribution_v1.json`

`results/trackvggt_scene_difficulty_v1.csv`

`results/trackvggt_dev5_retrospective_analysis_v1.json`

`manifests/trackvggt_full_audit_plan_v1.json`

如果合法创建新的 Cohort-V3，还需：

`manifests/trackvggt_expanded_training_v3.json`

`results/trackvggt_cohort_v2_v3_comparison.json`

所有 CSV 和 JSON 必须由原始真实审计记录生成。

不存在的指标写 NOT_MEASURED，不能写 0。

没有完成的序列写 INCOMPLETE，不能写 PASS。

---

# 15. 本次补充任务的 FINAL GOAL

**全面审计 VisionTrack 原始 TRAIN 的24条序列，并在严格场景和模型约束下，对12条关联训练序列完成真实 GMT 因果跟踪错误覆盖分析，确认当前 TrackVGGT 的训练样本是否不足，以及是否存在更大规模、有价值的跨摄像头身份纠错监督。**

最终至少回答：

1. 全部原始 TRAIN 的目标、身份、跨视角对应和目标尺寸分布是什么？
2. 12条关联训练序列中真实存在多少 GT 可确认的 GMT 身份关联错误？
3. 其中有多少可以通过改变候选身份排序或关联分数修正？
4. 正确身份候选的 Top-4/8/16 覆盖分别是多少？
5. 错误是否集中在少数场景、少数身份或小目标中？
6. 之前 1000/300 查询是否遗漏了大量有效困难事件？
7. 是否应该扩大到 Cohort-V3？
8. 扩大训练后应该保持哪些原有验证约束？
9. 新的数据分析是否改变 AG/QD 的研究优先级？
10. 目前的 VisionTrack 数据是否足以支撑具有泛化意义的结论？

**最重要：本次不要求训练新的大模型，也不要求重新开发新的跟踪架构。**

先完成真正的数据和错误空间审计。

如果审计证明存在足够多的可修正困难事件，则在原来的 TrackVGGT FINAL GOAL 下，继续执行扩大的 Identity-Guided AG、Track-Conditioned QD 与后续蒸馏训练。

如果发现可修正空间不足，也必须保留完整证据并解释限制，而不是再用简单样本反复尝试没有依据的网络改造。

不要因为某一个视频结果不佳就停止其他序列审计。

不要重新启动已经完成的合法计算。

不要无限扩大缓存和模型规模。

**现在立即开始。首先检查当前 TrackVGGT 最新状态与已完成的样本审计，然后建立全部24条 TRAIN 序列的审计清单，复用已有结果，补齐剩余关联训练序列的真实因果错误统计，最后形成是否扩大 AG/QD 训练样本的明确结论。**