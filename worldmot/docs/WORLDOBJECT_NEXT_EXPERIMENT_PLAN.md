# 批准后的 E0–E4 执行方案

状态：`WAITING_USER_APPROVAL`。以下全部是待执行命令，本轮没有运行。
不启动旧 `orchestrate_worldobject_six_gpu.py`；该入口在本分支已经退役。

## 授权与依赖

下一轮必须先收到用户明确“开始实验”，才可在仓库外生成授权文件。当前不存在此文件。
新文件 Schema 为 `worldobject_experiment_user_approval_v3`，必须包含 approved=true、
真实授权消息引用、当时准确 source_commit、`source_fingerprint()`、允许的 operations、splits 和 method_ids。
授权绑定代码 SHA；不接受旧实验授权、空文件或只修改 approved 字段。
只新增报告的 Git commit 不改变代码指纹，可保留已有授权与模型 Gate；代码/测试/方法配置字节改变则需要重新冻结。
本轮计划配置 `configs/worldobject_v3_codefix.json` 的 default_allowed=false，不是授权文件。

授权 operations 分别为 cache、freeze、train、replay、live、diagnostics、evaluate；不授权旧 schedule。
方法列表同时登记正式五种 V3 Method ID 和两个 LEGACY_CONTROL ID。不同方法必须有独立输出目录与 Manifest。

后续需要：有 Torch 的训练环境；真实 GMT Live 依赖原 Detectron2/GMT 环境；Ω512 的固定官方源码与权重；
官方 TrackEval、合法原始 GT 审计和 MATLAB/MEX。CPU 合成测试不证明这些真实依赖已经通过运行验证。
六卡可并行不同方法、分场景缓存任务，但训练选择仍使用单一 seed。训练流式读 cache，具体显存/内存需批准后测量。

准备的路径（执行前只读核对，不在本轮创建目录）：

```bash
WM_REPO=/data1/liuyeqiang/WWW_worldobject_codefix_20261025
WM_RUN=/data1/liuyeqiang/nas/A100/WorldMOT/worldobject_v3_after_approval_20261025
WM_APPROVAL="$WM_RUN/USER_APPROVAL.json"
WM_DATASET=/data/DATASETS/TRACKING/JDE/VisionTrack
WM_RAW_TRAIN=/data1/liuyeqiang/nas/A100/WorldMOT/gmt_detector_comparison_20261023/raw_train_v1
WM_RAW_TEST=/data1/liuyeqiang/nas/A100/WorldMOT/gmt_detector_comparison_20261023/raw_test_v1
WM_OMEGA_SOURCE=/data1/liuyeqiang/WWW_omega_runtime_20261016/official_vggt_omega
WM_OMEGA_WEIGHTS=/data1/liuyeqiang/WWW_omega_runtime_20261016/weights/vggt_omega_1b_512.pt
WM_AUDIT=/ABSOLUTE/PATH/TO/FROZEN_RAW_GT_AUDIT.json
export PYTHONPATH="$WM_REPO/worldmot"
```

`WM_AUDIT` 必须指向原有合法 GT 审计。不得创建修改过 GT 的“修复版”来提高结果。

## 最小缓存准备

只补 ROI：使用现有 raw GMT 与 v1 support 的只读联结，不重跑模型；可用于完整无几何方法。
完整几何：旧 sparse v1 不含相机深度修复所需密集信息，需在授权后新建 v2 目录。
如发现同源已保存的完整 depth/pose/pointmap，先核实 SHA，再决定是否可复用。
下面是无此密集资产时的重建命令，全部串行于各场景内部，不能把单场景时间打乱分到不同 GPU。

```bash
CUDA_VISIBLE_DEVICES=0 python "$WM_REPO/worldmot/scripts/cache_worldobject_support.py" \
  --authorization "$WM_APPROVAL" --dataset-root "$WM_DATASET" --split train \
  --raw-cache "$WM_RAW_TRAIN" --output "$WM_RUN/cache_train_v2" \
  --omega-source "$WM_OMEGA_SOURCE" --omega-weights "$WM_OMEGA_WEIGHTS" \
  --scenes 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 --device cuda:0

CUDA_VISIBLE_DEVICES=0 python "$WM_REPO/worldmot/scripts/cache_worldobject_support.py" \
  --authorization "$WM_APPROVAL" --dataset-root "$WM_DATASET" --split test \
  --raw-cache "$WM_RAW_TEST" --output "$WM_RUN/cache_test_v2" \
  --omega-source "$WM_OMEGA_SOURCE" --omega-weights "$WM_OMEGA_WEIGHTS" \
  --scenes 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 --device cuda:0
```

六卡提速时将场景列表分成互不重叠的六组，写入同一个新 v2 根目录的不同 scene 子目录；
必须先核实实际 TEST scene IDs 与冻结元数据一致。不要运行旧队列。失败的 v2 半成品不假装自动恢复 gauge。

```bash
python "$WM_REPO/worldmot/scripts/freeze_worldobject_scope.py" \
  --authorization "$WM_APPROVAL" --cache-train "$WM_RUN/cache_train_v2" \
  --raw-cache-train "$WM_RAW_TRAIN" --cache-test "$WM_RUN/cache_test_v2" --raw-cache-test "$WM_RAW_TEST" \
  --method-id FULL_WORLD_OBJECT_V3 --output "$WM_RUN/scope_FULL_WORLD_OBJECT_V3.json"
```

其他模式分别生成同名方法的 scope；No-Geometry 模式允许读取只读联结的 v1，不要求重算几何。
scope 的结构与各场景 Manifest SHA 都必须通过核验才能训练/回放。

## E0：身份特征恢复

冻结旧结果仍标为 source `190c735`。以下两个命令在相同新 v2 几何输入上运行固定旧架构，
唯一显式开关是 ROI 恢复；这是重新控制几何后的 feature attribution，不会复现旧坏深度的原始分数。

```bash
python "$WM_REPO/worldmot/scripts/run_world_object_legacy_control.py" \
  --authorization "$WM_APPROVAL" --cache "$WM_RUN/cache_train_v2" --raw-cache "$WM_RAW_TRAIN" \
  --shared-scope-lock "$WM_RUN/scope_FULL_WORLD_OBJECT_V3.json" --split train \
  --feature-mode legacy_disabled --output "$WM_RUN/E0_feature_off_train"

python "$WM_REPO/worldmot/scripts/run_world_object_legacy_control.py" \
  --authorization "$WM_APPROVAL" --cache "$WM_RUN/cache_train_v2" --raw-cache "$WM_RAW_TRAIN" \
  --shared-scope-lock "$WM_RUN/scope_FULL_WORLD_OBJECT_V3.json" --split train \
  --feature-mode restored --output "$WM_RUN/E0_roi_restored_train"
```

FEATURE_OFF 对照明确复现旧零视觉通路，只能作为该命名的失败对照。其回放计时不作为 Live E2E。
先使用 TRAIN 比较同人分裂、误合并和身份切换；TEST 对照只能在配置冻结后执行。

## E1：持续对象中心联合分配

与 E0 的 ROI-restored Pair-then-Merge 对照比较：

```bash
python "$WM_REPO/worldmot/scripts/run_world_object_fullvideo.py" \
  --authorization "$WM_APPROVAL" --cache "$WM_RUN/cache_train_v2" --raw-cache "$WM_RAW_TRAIN" \
  --scope-lock "$WM_RUN/scope_FULL_WORLD_OBJECT_V3.json" --split train \
  --method-id FULL_WORLD_OBJECT_V3 --rule-based --output "$WM_RUN/E1_joint_rule_train"
```

该比较同时改变联合分配和长期更新规则，必须登记为架构整体对照；不能把所有收益仅归因于求解器。
Primary V3 不调用 legacy builder，也不从 legacy 对照读取 ID。

## E2：真实身份监督训练

```bash
CUDA_VISIBLE_DEVICES=0 python "$WM_REPO/worldmot/scripts/train_world_object.py" \
  --authorization "$WM_APPROVAL" --cache "$WM_RUN/cache_train_v2" --raw-cache "$WM_RAW_TRAIN" \
  --scope-lock "$WM_RUN/scope_FULL_WORLD_OBJECT_V3.json" --dataset-root "$WM_DATASET" \
  --method-id FULL_WORLD_OBJECT_V3 --output "$WM_RUN/train_FULL_WORLD_OBJECT_V3" \
  --epochs 4 --seed 20261025 --hidden-dim 128 --clip-length 64 --lr 0.0002 \
  --history-mode predicted --device cuda:0
```

先18 fit/6 validation选 epoch，再重新初始化全24 TRAIN训练。checkpoint 未通过 v3完整 Gate 就不能进入 learned TEST。
训练期间只看 TRAIN 内部验证，不调 TEST。4 epoch 是初始有限预算；需根据实际训练损失/验证选择判断学习是否有效。

## E3：几何归因

四个独立 Method ID：Full、Head-Only No Geometry、Full No Geometry、Appearance Capacity-Matched。
同样 ROI、hidden_dim=128、seed、训练预算、生命周期和当前框输出。给每个方法生成对应 scope，再用 E2 命令替换
method-id/scope/output。可分别占GPU0–3；GPU4/5用于待批准的规则对照/分场景工作，禁止重启旧 V2/GMT 队列。
后两种方法在本版实现中具有相同架构容量和无几何通路，是分开登记的对照定义，不预设它们会产生不同结果。

## E4：完整 TEST、正式评测和速度

以下以 Full 为例，其他 learned 方法使用自己的合法 train_gate.json。

```bash
python "$WM_REPO/worldmot/scripts/run_world_object_fullvideo.py" \
  --authorization "$WM_APPROVAL" --cache "$WM_RUN/cache_test_v2" --raw-cache "$WM_RAW_TEST" \
  --scope-lock "$WM_RUN/scope_FULL_WORLD_OBJECT_V3.json" --split test --method-id FULL_WORLD_OBJECT_V3 \
  --checkpoint-manifest "$WM_RUN/train_FULL_WORLD_OBJECT_V3/train_gate.json" \
  --output "$WM_RUN/test_FULL_WORLD_OBJECT_V3" --device cpu

python "$WM_REPO/worldmot/scripts/evaluate_fullvideo.py" \
  --authorization "$WM_APPROVAL" --method-id FULL_WORLD_OBJECT_V3 --dataset-root "$WM_DATASET" \
  --split test --audit "$WM_AUDIT" --predictions "$WM_RUN/test_FULL_WORLD_OBJECT_V3" \
  --output "$WM_RUN/eval_FULL_WORLD_OBJECT_V3"

CUDA_VISIBLE_DEVICES=0 python "$WM_REPO/worldmot/scripts/run_world_object_live.py" \
  --authorization "$WM_APPROVAL" --dataset-root "$WM_DATASET" --raw-cache "$WM_RAW_TEST" \
  --scope-lock "$WM_RUN/scope_FULL_WORLD_OBJECT_V3.json" --split test --method-id FULL_WORLD_OBJECT_V3 \
  --checkpoint-manifest "$WM_RUN/train_FULL_WORLD_OBJECT_V3/train_gate.json" --detector-backend gmt \
  --omega-source "$WM_OMEGA_SOURCE" --omega-weights "$WM_OMEGA_WEIGHTS" \
  --output "$WM_RUN/live_FULL_WORLD_OBJECT_V3" --device cuda:0

python "$WM_REPO/worldmot/scripts/audit_persistent_world_identity.py" \
  --authorization "$WM_APPROVAL" --cache "$WM_RUN/cache_test_v2" --dataset-root "$WM_DATASET" --split test \
  --method-id FULL_WORLD_OBJECT_V3 --predictions "$WM_RUN/test_FULL_WORLD_OBJECT_V3" \
  --output "$WM_RUN/identity_diagnostics_FULL_WORLD_OBJECT_V3.json"

python "$WM_REPO/worldmot/scripts/summarize_worldobject_fps.py" \
  --predictions "$WM_RUN/live_FULL_WORLD_OBJECT_V3" --output "$WM_RUN/live_fps_FULL_WORLD_OBJECT_V3.csv"
```

正式 TEST 保留22/22视频、29,019时刻；14个合法 GT 场景评测，另外8个只保留完整推理并报告阻断原因。
不带 `--skip-matlab` 才会请求完整 MATLAB；未完成时状态是 `TRACKEVAL_COMPLETE_MATLAB_NOT_COMPLETE`，不能冒充全指标完成。
Live detector-backend=gmt 需要真实 GMT 环境；cache 模式单列 `RGB_CACHE_DETECTOR_OMEGA_TO_ID`，不能当成真实 GMT 前向 FPS。
E2E包含RGB读取、模型、对齐、关联、输出序列化与flush；记录P50/P95延迟和串行吞吐，不将六卡汇总吞吐充当单流FPS。

每次实际实验完成后，提交对应代码版本、真实报告与失败样本索引。任何无收益结果也完整保留。
本轮停在代码交付，不执行本文件中的任何真实数据命令。
