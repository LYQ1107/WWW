# CIR 当前方法与资格

当前真正启用的只有默认OFF和KEEP。原生五段compute/solve/propose/Bank/commit已抽取，保持原表达式、激活块、SciPy Hungarian、严格阈值、Bank和逐行提交。初始化不改变。全量Stage2 strict加载兼容，CIR没有新增nn.Module参数或随机初始化。

C1已接入原生提交前，并通过43项CPU合同与40组实际图像的C1官方/逐组路径一致性。原始分配先保留，C1改变分数时重用同一个完整原生Hungarian/严格阈值，最后仅一次执行原Bank/出生/字典提交。缺失可信源元数据时保持原分配。冻结alpha=2，使用原生合法候选及其严格过去的真实1152维descriptor版本；逐条L2，按ID求和再L2，与归一化query点积，直接加到原聚合分数。空历史增量0，同组已处理相机仍留作GTA候选，但不纳入C1严格历史。不会从GT创建目标。只作用正常主关联，第一组与Bank独立恢复器保持原始策略；Bank恢复后的新状态将由未来真实主关联读取。不是冻结状态重放。

FragmentLedger是已实现的有界metadata组件：每片段最多16条源键，每ID最多32片段，无视觉向量复制。连续(camera, nativeID)仅是观测关系假设，不是认证人物。当前只记录原生提交，关系版本必须0，不允许自行修改Global ID。已发布证据不能静默删除；内容hash链记录被保留部分的来源。其原生接入留待Phase2，CPU测试不作为完整状态修正资格。

OutputBuffer已实现0/2/4组FIFO发布、不可追改已发布帧及独立拷贝。尚未进入部署推理；默认W=0，只有TRAIN内部资格通过才启用窗口2/4。同样不把buffer内改输出ID视为真实原生修正。

StateTransaction将tracker所有状态、module Bank/sets和CIR状态一次deepcopy，保留跨根张量存储别名；恢复Python/NumPy/Torch及已初始化CUDA RNG，不为CPU审计初始化CUDA。真实40组中执行了一个丢弃图像/native步骤、恢复再运行，对后续输出/状态/RNG逐项相同。输出文件/外部观察器不在回滚所有权内，测试将被动来源字段保留、另开审计流，不把外部日志当作可回滚原生状态。

Guard、学习式风险头、原生SPLIT/RELINK尚未实现或启用，不把它们包装成已通过。待完整OFF通过后，先做C1实际图像在线和最小固定风险规则；规则正负都完整保留，按预注册资格继续独立必要机制。

新核心完整OFF仍在与已完成Original官方全视频比较；完整C1和DEV质量指标尚未运行。C1 GPU分数已与独立CPU逐候选公式核验（每边容差8*float32eps*(abs(expected)+3)，不修改分数或接受阈值），真实官方/逐组raw、stock、RNG和全部原生求解记录一致。
