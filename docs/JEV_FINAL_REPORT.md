# GMT/JEV Final Report（持续更新）

## 2026-10-05 protocol update

The canonical Stage2 checkpoint is now fixed to
`/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth`, SHA256
`cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8`, with
independent validation PASS. The historical `model_4500` artifact is proxy
screening only and cannot authorize official TEST.

Canonical OFF TEST reporting has been completed for this checkpoint (not used
for policy selection): TrackEval HOTA `8.3140`, DetA `6.4506`, AssA `11.9220`,
IDF1 `8.5297`, MOTA `-70.2413`, IDSW `3247`, Frag `60103`; cross-view audit
reports sequential CVIDF1/CVMA `8.1854/-70.2676` and interleaved
`8.1854/-70.9970`. Full provenance is in
`docs/STAGE2_OFF_TRACKING_RESULTS.md`.

The final report remains **IN PROGRESS**. A fresh strict same-GPU OFF gate,
formal TRAIN counterfactuals, H=1/8/16/32 three-seed policy-val selection,
validation-only calibration, reviewer controls, canonical selection lock, and
only then official TEST/Oracle/JEV evaluation are still required. No GO/NO-GO
claim is made here.

## 当前状态

截至 2026-10-05 UTC，项目已完成数据归档、完整性校验、GMT 环境安装、Detectron2 适配、VisionTrack JSON 生成和 JEV 源码/决策路径审计。Stage1 和 Stage2 均已完成；canonical `model_20000.pth` 已通过独立 checkpoint/reload 校验。正式 OFF baseline 已计算，但 formal v2 counterfactual、策略选择、selection lock 和官方 TEST 仍在推进。

因此本文档只记录已验证事实；OFF baseline 数值见顶部更新和 `STAGE2_OFF_TRACKING_RESULTS.md`，不把它们当作 JEV 的 GO/NO-GO 结论。

## 已验证事实

- VisionTrack `train.zip` 和 `test.zip` 均通过全包 CRC 校验；train/test 已解压到 `/data/DATASETS/TRACKING/JDE/VisionTrack`。
- train：48 个序列、57,508 张图、596,749 条 JSON 标注；test：44 个序列、58,038 张图、580,178 条 JSON 标注。
- `00012night_View2/gt/View2.txt` 转换出的标准 MOT `gt.txt` 与仓库既有 TrackEval `vision-train` 版本逐行一致；原始文本仍保留。
- `ids_range/len_seq` 只作为 loader 兼容字段添加，官方 converter 输出已备份，annotations SHA256 见 `manifests/loader_metadata_adaptation.json`。
- GMT loader 实际加载 `VISION_stage1`、`VISION_train`、`VISION_test` 成功；模型/mapper 单卡 forward 和 8-GPU canonical smoke 的 loss 均 finite。
- 原始 backbone checkpoint 直接加载会丢失 DFConv wrapper 的两个 key 并在训练 smoke 中产生 NaN；`CH_FPN_1x_key_adapted.pth` 只改 key、不改 tensor 数值，适配映射和 SHA256 见 `manifests/backbone_key_adaptation.json`。
- 10 个 JEV 开源仓库已按核心源码审计，结论见 `JEV_OPEN_SOURCE_AUDIT.md`。
- 已验证 typed JEV controller、online runtime/commit boundary、frozen-evidence mutable GMT-state counterfactual runner、状态 schema、baseline family、离线 trainer、NLL/Brier/ECE/risk-coverage 指标和结构性 stress tests；对应 CPU invariant tests 全部通过。这些是接口/防泄漏证据，不是 tracking 结果。
- 结构性压力测试已保存至 `outputs/jev_stress_contract.json` 并 PASS；项目完整回归套件在正确 `PYTHONPATH` 下 16/16 PASS。该结果不替代真实冻结 Stage2 的 tracking/counterfactual 结果。
- VisionTrack 空预测评测回归已覆盖全部 44 个 test 序列；原始 GT 未改动，并审计记录了重复 ID 行。严格唯一 ID 检查与显式 permissive 审计模式均有区分，尚未产生正式 GMT tracking 指标。
- Stage1 最终 checkpoint 为 `outputs/stage1_single_gpu/model_16000.pth`；Stage2 canonical checkpoint 为 `outputs/stage2_single_gpu/model_20000.pth`，SHA256 `cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8`。Stage2 iteration/scheduler/global=20000、411 个模型 key、optimizer state、finite 参数和 reload 均通过。

## 正式训练约定

当前正式延续运行使用：

- `configs/VISION_stage1.yaml` 的 canonical 20,000 global iter、REID=True、TRAIN_LEN=10、TRAIN_SIZE=1280；
- GPU0 单卡、`SOLVER.IMS_PER_BATCH=1`，Stage1 输出目录为 `outputs/stage1_single_gpu`；
- `CH_FPN_1x_key_adapted.pth`；
- activation checkpoint；
- Gloo/CPU-staged collectives；
- Stage2 同样固定在 GPU0 单卡、batch size 1，并从 Stage1 最终 checkpoint 开始。

GPU 数量和有效 global batch 与原始 10-GPU 运行环境存在记录在案的硬件差异；单卡路径是本机测得的最快稳定路径，不应被表述为 bit-for-bit 官方复现。GPU1 的 12 GiB 外部残留上下文未纳入任务；Stage1/Stage2 gate 会报告这些差异。

## 尚未填写的结果

以下内容必须等冻结 Stage1/Stage2 和 counterfactual/oracle 数据后填写：

1. OFF/Oracle/threshold/state-threshold/MLP/JEV 的共享 checkpoint 结果；
2. action accuracy、NLL、Brier、ECE、risk-coverage；
3. wrong commits、contamination duration、secondary IDSW、stale reactivation、recovery latency；
4. HOTA、DetA、AssA、IDF1、MOTA、IDSW、Frag、CVIDF1、CVMA；
5. 参数量、GPU memory、latency、额外 GMT calls；
6. 依据 oracle headroom 和 matched baselines 的 GO/NO-GO。

## 当前结论

当前仍只能结论为 **IN PROGRESS**。Stage2、真实 counterfactual/oracle、正式 baseline/JEV 指标和 GO/NO-GO 审计完成前，不声称 JEV 提升或论文数值复现。
