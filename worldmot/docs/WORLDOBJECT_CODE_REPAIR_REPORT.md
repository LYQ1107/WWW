# WorldMOT 世界对象架构代码修复验收

状态：`CODE_REPAIR_COMPLETE_WAITING_USER_APPROVAL`。

源分支：`research/worldmot-worldobject-fast-proof-20261025`。
源提交：`190c735957f1079ba3f770c8aa35a341d72ba60a`。
独立修复分支：`research/worldmot-worldobject-codefix-20261025`。
测试过的实现提交：`41c023d7c675d529b4fe794e5b64afafe757b938`。
GitHub发布的实现提交：`e499b83456a015f2df00edb76c6bbc2a12767bf0`。
两者完整文件树均为 `f7f719c3edab3fe98f46e92403d920c69ca98e25`，代码与测试字节完全相同。
最终交付提交为包含本文和审计 JSON 的分支 HEAD；最终回复返回其真实 commit 链接。
CPU 报告登记测试实现 commit 与完整代码指纹，避免报告自包含 Git hash 的循环依赖。
本机HTTPS Git没有写入凭据、SSH22端口超时，因此使用已连接的GitHub Git Data接口发布。
API提交元数据与本地不同；先核对相同Git tree SHA，再保留本地提交于
`archive/worldobject-codefix-local-20261025`，将交付分支同步到远端提交；不会改动工作文件。

旧 worktree 当时本地 HEAD 为 da59743，带有未提交报告/脚本；本轮没有 stash、覆盖或提交这些内容。
新 worktree 直接从核实过的远端190c735建立。旧 GMT/V1/V2 模型、结果、原始 GT 和 NAS 资产均未修改。
修改文件清单见 `results/worldobject_code_repair_audit.json:modified_files`。

## 原始问题、机制与真实 CPU 验证

旧位置均指源提交190c735。新位置按修复分支的仓库相对路径给出。

| 原始问题 | 具体旧代码 | 修复文件 | 修复机制 | CPU 验证 | 状态 |
|---|---|---|---|---|---|
| 身份特征缺失 | `scripts/cache_worldobject_support.py:112`未保存ROI；`scripts/run_world_object_fullvideo.py:99`使用score/quality；`models/multi_view_world_object.py:15`无feature | `models/world_object_observation.py`、`data/worldobject_feature_join.py`、`data/worldobject_cache.py`、缓存producer | 原维真实ROI、Manifest维度校验、只读精确联结、缺失时报错；Hypothesis显式携带feature | `test_roi_feature_survives_observation_assignment_and_bank`；`test_v2_producer_reader_feature_and_camera_contract` | PASS |
| 世界z冒充相机深度 | `models/world_object_support.py:160` | `models/world_object_support.py`、`geometry/world_frame_contract.py`、`geometry/camera_model.py` | Camera-from-Local-World获取前向深度；Sim(3)协方差与当前时间契约 | `test_rotated_camera_clusters_camera_depth_not_world_z`；`test_camera_pose_translation_scale_and_sim3_covariance` | PASS |
| Greedy预合并 | `models/multi_view_world_object.py:56` | `tracking/joint_world_object_assignment.py`、`runtime/world_object_tracking_core.py` | 历史对象直接竞争单/双视角观测、NULL和Birth；小分量精确/大分量有限Beam | `test_conflicting_pair_competes_with_consistent_historical_option`；`test_large_component_is_deterministic_and_bounded` | PASS |
| 历史身份污染 | `tracking/persistent_world_object_bank.py:54`默认零向量、`:129`直接覆盖 | `tracking/persistent_world_object_bank.py`、`models/multi_view_world_object.py` | 分开管理位置和身份；只更新真实匹配、合格置信度的ROI，门控EMA与Memory Delta | `test_unmatched_and_low_confidence_features_do_not_rewrite_other_identity`；`test_update_gate_and_memory_delta_are_used_and_read_next_step` | PASS |
| W4监督与实际关联无关 | `scripts/train_world_object.py:105`只有gate/birth BCE；`run_world_object_fullvideo.py:119`最近中心选memory | 新Identity Head、`training/world_object_identity_targets.py`、`world_object_assignment_loss.py`、`world_object_causal_training.py` | 真实M×N身份分配；共享ID/误合并/生命周期/状态一致性；同一因果Bank | `test_assignment_loss_has_history_specific_gradient_and_memory_gradient`；`test_assignment_logits_change_the_selected_historical_id` | PASS |
| W5消融错误 | 旧 `train_world_object.py:feature()` 与 `FrozenUpdateHead`只清空7维几何 | `tracking/world_object_config.py`、输入构造器、Live适配器和Bank | 四个独立主对照Method ID；Full No Geometry禁用上游到更新的全部几何作用 | `test_entire_no_geometry_path_is_invariant_to_arbitrary_geometry`；`test_head_only_ablation_retains_upstream_spatial_decisions` | PASS |
| 几何失效仍混入局部点 | `runtime/online_world_object_tracker.py:57`取历史Transform，未对齐support仍送Bank | `runtime/online_world_object_tracker.py`、`data/worldobject_observations.py` | 当前无效几何只走真实视觉回退；不写入错坐标，拒绝旧时间Transform | `test_alignment_failure_in_live_frontend_does_not_reuse_transform`；`test_stale_world_transform_is_rejected_even_if_marked_valid` | PASS |
| Live/Replay核心分叉 | 旧Live `:75`用ROI/score；Replay `:99`用两维分数，分别组装决策 | `runtime/world_object_tracking_core.py`、两个runner | 同一Observation、同一身份核心和输出协议，当前框原样输出 | `test_live_and_replay_adapters_produce_identical_core_ids_boxes_and_state` | PASS |
| TEST目标/未来泄漏边界不足 | 旧cache reader直接返回identity数组；重复图像键可覆盖 | `data/worldobject_cache.py`、`data/visiontrack.py`、适配器、Bank | 隔离target字段、metadata-only数据、拒绝非因果时间与重复键、场景重置 | `test_inference_adapter_refuses_target_fields`；`test_invalid_time_and_duplicate_inputs_fail_before_state_mutation` | PASS |
| 实验入口无批准闸门 | 旧train/cache/replay/live/evaluate和六卡调度入口 | `runtime/experiment_authorization.py`及相应CLI | 默认拒绝；操作/方法/split/代码指纹绑定；旧六卡队列退役 | `test_real_entrypoints_deny_without_approval_before_opening_resources` | PASS |

上述位置省略统一前缀 `worldmot/worldmot/` 或 `worldmot/`；完整路径见修改清单。

附加修复：源版本 `scripts/summarize_worldobject_fps.py` 实际只有一行 shell 错误文本，已替换为合法的
只读计时日志汇总器。模型/运行时/跟踪包改成延迟导出，保留旧公共API，防止纯数据契约立即加载旧V2网络。
Git worktree来源信息使用只读ref解析，避免初始化Torch后为获取commit再fork；CPU检查使用GNU MKL线程层。

## CPU 验收事实

最新报告：`results/worldobject_cpu_test_report.json`。

- 79项合成测试通过，0失败，0跳过；pytest exit code=0。
- AST：180个Python文件，0语法错误。
- Pyflakes：57个变更Python文件，0问题；`git diff --check`通过。
- 单独导入models/runtime/tracking公共包与授权模块：Torch未初始化，检查通过。
- 测试覆盖用户A–O情形，包括120个密集合成候选、资源上限回退、空帧、单人、多视角遮挡、非有限深度和重复键。
- 测试中的8×8图像、ROI、小型NPZ、GT及梯度全部合成，未打开真实VisionTrack、GMT/Ω预训练权重。

真实修复过程也记录失败：初次特征联结测试暴露 `PersonDetection.view_id` 与 camera_id 不一致，已修复；
初次AST检查发现继承的FPS脚本损坏，已修复；混合OpenMP环境出现过进程终止，使用GNU线程层与延迟导入后
最终完整检查成功。不设置 `KMP_DUPLICATE_LIB_OK`，不把失败运行写为PASS。

## 新旧资产与启动条件

| 资产/接口 | 当前结论 |
|---|---|
| GMT原始候选、ROI、bbox、score | 设计为只读恢复；真实联结需要批准后的实际数据检查 |
| 旧Ω sparse v1 | 可只读用于特征恢复；几何标为未验证，不能直接用于完整V3几何 |
| 新support v2 | producer/reader和SHA契约已实现、合成闭环通过；真实缓存未生成 |
| 旧W4/W5 checkpoint | Schema和语义不兼容，禁止加载到V3继续声称有效 |
| 新Identity模型 | 训练/验证/完整Gate代码已实现；本轮没有训练出的V3 checkpoint |
| 正式TEST | learned路线需要新的完整TRAIN Gate；rule-based仅作为显式非学习对照 |
| E0/E1 | 独立legacy控制入口已准备，不进入正式核心；E1属于架构整体对照 |
| E2/E3/E4 | 命令、方法ID与协议已准备；尚未获得执行授权 |

当前没有未修复的P0代码阻断。真实启动前仍需明确批准、核实模型环境、准备新几何支持并完成新身份模型训练。
v2缓存断点缺少完整对齐快照时采取拒绝恢复；实际显存/延迟、真实分布收敛和指标收益均未验证。
因此“代码修复完成”只表示上述接口与CPU验收通过，不表示HOTA/IDF1或FPS有所提高。

## 其他交付

- [架构与生命周期](WORLDOBJECT_ARCHITECTURE_V3.md)
- [特征、坐标与最小迁移字段](WORLDOBJECT_FEATURE_AND_GEOMETRY_CONTRACT.md)
- [联合分配目标、约束和复杂度](WORLDOBJECT_JOINT_ASSIGNMENT_DESIGN.md)
- [Tensor形状、损失和监督](WORLDOBJECT_TRAINING_LOSS_AND_SUPERVISION.md)
- [批准后E0–E4命令](WORLDOBJECT_NEXT_EXPERIMENT_PLAN.md)

代码修复阶段已经结束。目前没有执行任何新的真实训练、推理或评测。
下一阶段六卡全量实验尚未启动，等待用户明确批准。
