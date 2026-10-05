# GMT/VisionTrack 当前诊断报告与后续计划

**报告状态：`BASELINE_DIAGNOSIS_IN_PROGRESS`**<br>
**生成时间：2026-10-05（UTC）**<br>
**审查分支：`jev/reviewer-proof-v2`**<br>
**报告基线提交：`4ace888`**

## 结论摘要

当前最有证据支持的根因是 **VisionTrack 多视角输出顺序与 evaluator 图像标签顺序不一致**，而不是已经证明的 JEV 效果或模型权重失效。

证据链已经完成到“诊断恢复”阶段：

- 原始 canonical OFF 评测的 HOTA `8.314` 等异常低值不能作为论文结果，也不能用于宣称 JEV 提升。
- 在固定 TRAIN mini 子集和完整 TRAIN 上，按源码顺序契约进行确定性 view-block → frame-major 标签校正后，检测框 IoU 和召回率恢复到合理范围。
- 用同一校正生成的 TEST 诊断副本得到 HOTA `67.442`、IDF1 `82.239`、MOTA `80.942`；这些是诊断值，不是最终官方值。
- COCO→MOT round-trip 全量检查 PASS，因此数值格式转换不是主要根因。
- strict same-GPU OFF traced 仍在运行；正式 evaluator 修复和源码回归尚未完成，所以 baseline 尚未正式解锁，JEV policy selection、`FINAL_SELECTION_LOCK` 和官方 TEST 仍然保持阻塞。

## 已完成、可复核的证据

### 1. 固定输入与 checkpoint

| 项目 | 值 |
| --- | --- |
| canonical Stage2 | `/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth` |
| Stage2 SHA256 | `cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8` |
| Stage1 | `/data1/liuyeqiang/WWW/outputs/stage1_single_gpu/model_16000.pth` |
| Stage1 SHA256 | `143e84deb50bdf5379c8f4463f1b9b237132e9281726b9cff469aff8c9dbe64e` |
| inference config | `/data1/liuyeqiang/WWW_jev_v2/configs/VISION_test.yaml` |
| config SHA256 | `bdaeca71d875e8824c3eaf825967a7ba032514297642a6aabe7ad10a740be94a` |
| TEST annotation SHA256 | `7a1e735bf25026a56148deaeb432e0520c6ad5dad97d5362aa58739255389593` |
| TRAIN annotation SHA256 | `9093c36204bae482c2f74997f531bfe2d8a46464f4541093703ef22db0068c85` |

Stage1 的最终文件名是 `model_16000.pth`，其 scheduler 到达全局 iteration 20000；Stage2 独立完成 20000 iterations。两阶段 metrics 中记录的 loss 均为有限值，最终 checkpoint 可加载。为满足 `/data1` 的 formal resource gate，已将不再作为 canonical 输入的中间 checkpoint 移到可恢复目录：

`/home/liuyeqiang/checkpoint_archive_20261005`

原位只保留上述两个关键 checkpoint，未删除恢复数据。

### 2. 原始异常值（仅作为 incident 记录）

原始 canonical TEST prediction：

`/data1/liuyeqiang/WWW/outputs/research_final_v2/off/inference_test/inference_VISION_test/coco_instances_results.json`

SHA256：`a8d2f5c4cc758a887276e59671831215d36a8949d07486a34d78076aaf3e7b3d`

| Metric | 原始诊断值 |
| --- | ---: |
| HOTA | 8.3140 |
| DetA | 6.4506 |
| AssA | 11.9220 |
| IDF1 | 8.5297 |
| MOTA | -70.2413 |
| IDSW | 3,247 |
| Frag | 60,103 |
| prediction rows | 527,474 |
| GT detections | 580,178 |
| CVIDF1 | 8.1854 |

这些值被保留用于追溯，但已标记为 baseline incident，不得当作最终结果。

### 3. 输出顺序根因证据

源码契约为：

1. `GMTDatasetMapper` 产生 `[view1 的全部帧, view2 的全部帧, ...]` 的 view-block 输入。
2. `GTRRCNN.sliding_inference_GMT` 内部按帧处理多视角，并在返回前构造 `[frame1 的全部 view, frame2 的全部 view, ...]` 的 frame-major 输出。
3. 现有 `MOTEvaluator.process` 原先直接 `zip(inputs, outputs)`，导致输出被绑定到错误的 `image_id`。

这个顺序差异已经在初始官方源码提交
`8fa96705e86f74dcc880b1cf332e8e448ac3ef0f` 中存在，因此目前证据指向 evaluator/source mapping bug，而不是 JEV 新增逻辑单独造成的回归。

固定 TRAIN mini（scene `00002garden`、`00004garden`，两视角，5,636 images）结果：

| 诊断量 | 未校正 | 确定性顺序校正后 |
| --- | ---: | ---: |
| mean max IoU | 0.0945 | 0.8243 |
| median max IoU | 0.0000 | 0.8746 |
| Recall@IoU 0.5 | 5.39% | 95.57% |
| Precision@IoU 0.5 | 5.55% | 98.39% |
| TP / FP / FN | 991 / 16,856 / 17,402 | 17,560 / 287 / 833 |

完整 TRAIN 结果：

| 诊断量 | 未校正 | 确定性顺序校正后 |
| --- | ---: | ---: |
| mean max IoU | 0.1258 | 0.7888 |
| median max IoU | 0.0000 | 0.8415 |
| Recall@IoU 0.5 | 10.15% | 89.91% |
| Precision@IoU 0.5 | 11.11% | 98.46% |
| F1@IoU 0.5 | 10.61% | 93.99% |
| TP / FP / FN | 60,553 / 484,351 / 536,196 | 536,530 / 8,374 / 60,219 |

完整 TRAIN 的 24 个双视角视频全部报告
`FRAME_MAJOR_OUTPUT_VS_VIEW_BLOCK_INPUT`；证据文件：

`/data1/liuyeqiang/WWW/outputs/research_final_v2/baseline_debug/output_order_train.json`

SHA256：`07ce57732fe43a52d13fd5c2e9bc235c0e260a5b9074415fdbb979baafdc9816`

### 4. 校正 TEST 诊断副本

输入没有被覆盖；校正副本为：

`/data1/liuyeqiang/WWW/outputs/research_final_v2/baseline_debug/canonical_test_predictions_order_fixed.json`

该副本改变了 527,026 个 prediction rows 和 57,994 个 image labels，SHA256：
`263f6aaedbac159e6c3c4a2703dd31b0ca0d24bfe7f351bd0807872c5af57903`。

在现有 permissive TrackEval 诊断转换下：

| Metric | 校正诊断值 |
| --- | ---: |
| HOTA | 67.442 |
| DetA | 66.278 |
| AssA | 68.992 |
| IDF1 | 82.239 |
| MOTA | 80.942 |
| IDSW | 3,092 |
| Frag | 8,004 |
| CVIDF1 | 79.0248 |
| sequential CVMA | 80.9276 |
| interleaved CVMA | 74.2925 |

对应证据：

- TrackEval metrics：`/data1/liuyeqiang/WWW/outputs/research_final_v2/baseline_debug/corrected_test_eval/evaluation/metrics.json`，SHA256 `dfb5e83f96f3e70fc412938ab7db694555c39efdf5d6536da52a890882e11041`。
- Cross-view metrics：`/data1/liuyeqiang/WWW/outputs/research_final_v2/baseline_debug/corrected_test_eval/crossview/metrics.json`，SHA256 `ef9c3ce4cdbe992fd012aa11b2be0b34e660c751f3845d2d2230de6812f789e5`。

### 5. COCO/MOT round-trip

全量比较 527,474 rows，另有固定 seed 的 1,000 rows sample：

- unknown image IDs：0
- missing sequences：0
- mismatch count：0
- 最大 bbox 绝对误差：`5.0e-7`
- 最大 score 绝对误差：`5.0e-7`

证据文件：
`/data1/liuyeqiang/WWW/outputs/research_final_v2/baseline_debug/coco_mot_roundtrip.json`

SHA256：`0412381b261a07d7bfd69cf0e30abcbef8e440d55afc4518b2f0eb3736937bae`

因此，COCO→MOT 数值转换不是原始异常的主要解释；这项 PASS 不等于正式 evaluator 修复已经完成。

## 当前运行状态

| 项目 | 状态 |
| --- | --- |
| strict traced OFF | RUNNING，GPU0，PID 17269 |
| strict 父进程 | PID 17243；正式 pipeline PID 16955 |
| 报告生成时进度 | 约 12/22 个视频批次完成，第 13 批次约 50% |
| checkpoint | 无训练进程；Stage1/Stage2 已完成并保留最终 checkpoint |
| `/data1` 空间 | 约 33 GB free，已满足当前 30 GB resource gate |
| 当前 JEV policy selection | HOLD |
| `FINAL_SELECTION_LOCK` | 尚未生成，按协议保持阻塞 |
| 官方 TEST JEV 结果 | 尚未生成，按协议保持阻塞 |

strict traced 使用的输入仍是旧 source commit `f3187b8` 的完整 OFF 运行；它必须先自然完成，不能中途终止或改动其运行源树。

## 尚未完成任务与执行计划

以下项目不是“已完成结果”，而是交给后续执行和审查的明确清单：

1. **完成 strict OFF 证据链。** 等 native/traced 收尾，检查 same-GPU OFF exact trajectory、4A trace contract、4B simulator replay；保存各自 manifest、trace 和 hash。失败则只定位并修复失败 gate，不跳过。
2. **合入正式 evaluator 顺序修复。** 已在隔离 worktree 准备最小修复和单测（本地准备 commit `8dfc9e4`，尚未合入正式 worktree/远端），其行为是仅对 `VISION_train`/`VISION_test` 将 evaluator 输入按 view-block 重排为 frame-major，保留单视角和其他数据集行为。
3. **做源码回归。** 用固定 mini subset、同一 checkpoint/config，比较修复前后的 image_id 映射、bbox/score/track_id；确认 JEV-disabled 输出仅发生预期的 evaluator label correction，没有模型/轨迹算法变化。再运行修复后的正式 OFF TEST（必要时补 TRAIN）到新的 diagnostic namespace，绝不覆盖旧 anomaly artifact。
4. **完成 baseline root-cause report。** 将源码回归、checkpoint load、config diff、training exposure、修复后正式 OFF 指标合并到 `docs/GMT_BASELINE_ROOT_CAUSE_REPORT.md` 和 `DIAGNOSIS.json`；只有证据链闭合后才把 baseline 标记为 recovered。
5. **恢复 formal counterfactual TRAIN 数据。** 先生成并验证 TRAIN H=8，再生成 H=32 并派生 H=1/8/16/32；每个 shard 串行运行、保存 manifest/hash，校验 sample coverage、future-GT 禁止条件、OFF replay 一致性和磁盘/内存 gate。旧的并发 shard 失败不能当作 PASS。
6. **完成 JEV 决策模块和 policy validation。** 在 baseline 解锁后，按固定 TRAIN-only split、固定 seed 集合、equal supervision 训练/比较候选 policy，包含 threshold/MLP controls、Oracle 对照、stress test 和 horizon 对照；禁止用 TEST 选择超参或挑 best seed。
7. **生成并验证 `FINAL_SELECTION_LOCK`。** 锁定 canonical `model_20000.pth`、policy/horizon/seed、selection manifest 和 hashes；锁前不读 TEST GT 做选择，锁无效时不生成 official TEST counterfactual。
8. **最后才做官方 TEST 与最终报告。** 生成 official counterfactual、OFF/Oracle/control/JEV 统一评测、cross-view 指标、failure decomposition、reproducibility manifest 和 `FINAL_REPORT.json`；所有结论必须引用可验证的 artifact/hash。
9. **推送最终代码和结果。** 将 evaluator 修复、root-cause report、selection lock、正式 metrics、运行说明和失败/重跑记录提交到此分支；`results/` 等用户未跟踪目录不加入 Git。

## 审查边界

- 当前校正 TEST 数字只证明“顺序错配可以解释原始异常”，不证明 JEV 已有效。
- 当前不允许把 `67.442 HOTA` 等诊断副本当作官方论文结果。
- 当前不允许跳过 pre-JEV/source regression、正式 evaluator 修复、selection lock 或 TEST isolation。
- 旧服务器 checkpoint 不在审查范围；只使用本报告列出的 canonical Stage1/Stage2、当前 repository/config、当前预测、trace 和 cache。
