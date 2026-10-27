WorldMOT V5：固定TRAIN6历史复用快照审计注册

2026-10-27T07:05:02.658100+00:00，尚未执行本次审计。

现有训练日志中的 wrong_world_object_merges 只统计当前时刻同一ID分配给不同GT人物，不能替代错误历史ID复用事件。mixed_historical_objects_before 可反映历史混合暴露，但不是错误复用次数。本次补充一次有限的TRAIN诊断，保留当前正常训练与共同12轮预算。

固定使用B3的一个冻结中间恢复快照，在TRAIN Scene 4/8/12/16/20/24上，各从视频起点连续执行，共5,860个同步时刻、172,760个原始检测候选。只使用1个CPU线程，不占用正在训练的GPU，不创建优化器、不改模型参数或关联阈值。

快照选择规则是注册时第一个可读取并通过SHA校验的完整Scene恢复点，不按VAL、TEST或本次审计分数选择。实际快照完成2个最终训练epoch，处于第3轮，已完成场景为[11, 20, 19, 6, 2, 9, 16, 17, 10, 1, 3, 7, 12]。

恢复文件SHA-256：`e888fc1d347005b8d78fdf9c608b396f29036ad27987a6eb09c48d1c49266f2e`。

模型张量SHA-256：`1cfbadd019409b59f643a56f994a66ac7c4d72a3b62186178b4e7d50985d9851`。

快照已复制到独立只读文件，并在真实挂载的NAS完成完整读回SHA校验。它不是最终12轮权重，没有TEST资格。

逐时刻先完成模型决策，再读取TRAIN标注作诊断；GT不修正历史、不影响下一时刻的模型输入。复用已核验的DeliveredIdentityAudit口径，统计错误历史复用、累计混合ID、False Birth、同时Duplicate Birth、NULL、LOST条件恢复和检测间隔保ID，并保留计数、分母和有限失败例子。缺失输出不能成为正确跨视角事件。

所有场景都已被该快照训练见过。本次结果是冻结快照在TRAIN上的自主闭环诊断，不是独立VAL、正式TrackEval/CVIDF1或TEST结果；也不是原训练过程中每一步权重与随机状态的重新构造。单个快照没有方法比较或几何独立收益结论，不用于模型、参数、预算或checkpoint选择。

启动前必须存在GitHub分支引用、树和公开fetch核验通过的注册回执；源码、注册、快照和原始TRAIN标注SHA在运行中保持一致。只能在完整6场、全部时刻与候选覆盖、模型张量未变的条件下标记审计完成。

端到端耗时包含GT连接、审计与输出，不能作为已注册的缓存/真实RGB推理FPS比较。正式B2/B3/B4 TEST继续要求三模型各自完整24场×12轮Gate，不改原有锁或正式评测范围。

- [固定注册、输入SHA和快照进度](../results/v5_train_snapshot_audit_registration.json)
- [5项生成数据边界检查](../results/v5_train_snapshot_audit_checks.json)
- [只读CPU审计入口](../analysis/audit_worldobject_v5_train_snapshot.py)

本次不会改写V5正式指标表；审计完成后另存TRAIN快照报告。

执行结果另见 [完整TRAIN6快照审计结果](V5_TRAIN_SNAPSHOT_HISTORY_REUSE_RESULTS.md)。以上注册参数保持不变。
