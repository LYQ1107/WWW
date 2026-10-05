# WWW / GMT-JEV 科研任务运行说明书

更新时间：2026-10-05 07:39 UTC（运行状态是快照；PID 和百分比会变化）。

这份文档说明当前在跑什么、后续会跑什么、每一步的完成条件和保守估时。它是运行手册，不是最终实验结论；在 `PIPELINE_COMPLETE.json`、`FINAL_REPORT.json` 和最终锁都出现并通过校验前，不得宣称 JEV 有效或 GO。

## 1. 固定的实验基准

- canonical 仓库：`/data1/liuyeqiang/WWW`
- 当前代码 worktree：`/data1/liuyeqiang/WWW_jev_v2`
- 分支：`jev/reviewer-proof-v2`
- canonical Stage2 checkpoint：`/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth`
- checkpoint SHA256：`cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8`
- Stage2 checkpoint 校验：`PASS`（iteration、scheduler/global iteration 均为 20000；optimizer、finite、reload 均通过）
- 配置：`/data1/liuyeqiang/WWW_jev_v2/configs/VISION_test.yaml`
- 配置 SHA256：`bdaeca71d875e8824c3eaf825967a7ba032514297642a6aabe7ad10a740be94a`

checkpoint、VisionTrack 数据、推理输出和训练结果不提交 Git。worktree 中已有的未跟踪 `results/` 属于用户内容，必须保留但不纳入本说明书提交。

## 2. 当前正在运行的任务（快照）

| PID | 任务 | 当前作用 | 备注 |
|---:|---|---|---|
| 16606 → 16673 → 16699 | strict same-GPU OFF gate | GPU0 上顺序执行 native OFF；完成后才执行 traced OFF | 正式 replay 的前置 gate |
| 10350 | formal TRAIN counterfactual shard 0 | 视频 `1 5 9 13 17 21`，canonical worktree | 输出完成前不合并 |
| 29280 | formal TRAIN counterfactual shard 1 | 视频 `2 6 10 14 18 22`，低内存 legacy worktree | 结果仍绑定 canonical checkpoint |
| 33788 | pre-lock TEST counterfactual diagnostic | 视频 `1 5 9 13 17 21` | 已降级为隔离诊断；`formal/test/DO_NOT_USE_FOR_SELECTION`；绝不用于 selection/final report |
| 10016 | 原先启动的 pipeline parent | 等待/管理 formal 阶段 | 日志尾部可能包含旧失败记录，以当前 PID 和 manifest 为准 |

2026-10-05 07:39 UTC 快照中，strict native 仍为 `RUNNING`。注意日志中的 `1040/1040`、`1904/1904`、`1001/1001` 是单个视频组内部的进度，不是整个测试集的进度。当前日志尾部视频组为 `65/1001` 步；prediction stream 已写入 `49,514/58,038` 张图（约 85.3%，当前组结束后才会更新）。GPU0 正在工作，`/data1` 剩余空间约 47 GB；native 完成前不算 gate 完成。formal 目录尚无可合并的完整 shard manifest。

查看实时状态：

```bash
cd /data1/liuyeqiang/WWW_jev_v2
ps -eo pid,ppid,stat,etime,%cpu,%mem,rss,cmd | rg \
  'run_final_v2_pipeline|run_formal_inference|run_isolated_test_net|build_jev_counterfactual_v2'
tail -c 10000 /data1/liuyeqiang/WWW/outputs/research_final_v2/off/same_gpu_strict/native/inference.log \
  | tr '\r' '\n' | rg -o '[0-9]+/[0-9]+ \[[^]]+\]|[0-9]{1,3}%' | tail
jq . /data1/liuyeqiang/WWW/outputs/research_final_v2/off/same_gpu_strict/native/inference_manifest.json
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
df -h /data1
```

不要因为日志暂时没有新文件就杀进程：counterfactual builder 主要在内存中构建记录，通常到完整 shard 结束时才写 JSONL 和 manifest。

## 3. 后续执行顺序

流水线必须按以下顺序推进，不能跳过 gate：

1. **完成 strict native OFF**：GPU0 上的 native 结果必须写出并在 manifest 中为 `COMPLETE`。
2. **完成 strict traced OFF**：仍使用可见 GPU0、相同 checkpoint/config，写出 `traced_off.jsonl` 和 traced prediction JSON。
3. **严格同卡 OFF trajectory gate**：native/traced 必须在同一可见 GPU 顺序执行；`image_id/category_id/track_id`、逐帧 assignment、new-ID/reactivation/action sequence 必须 exact；bbox、score、trace feature/probability 只允许 `atol=rtol=1e-6`。`raw_json_equal` 只保留为 audit 字段。报告为 `off/same_gpu_off_gate.json`。
4A. **TRACE_OFF_CONTRACT_GATE**：验证 trace 中每个事件的 `off_action == proposed_action == committed_action`、one-hot 概率、state digest、无 future-GT 访问；报告为 `off/trace_off_contract_gate.json`。
4B. **V2_SIMULATOR_OFF_REPLAY_GATE**：用同一 checkpoint/config、冻结 perception cache 和 formal GMT association adapter，完整重放“无干预 OFF”轨迹；比较 proposal、Hungarian assignment、track ID、memory write/length、stale bank、reactivation/new-ID 和最终轨迹 digest。报告为 `off/v2_simulator_off_replay_gate.json`。只有 4B PASS 才允许正式 policy labels。
5. **完成 TRAIN formal counterfactual**：先保留当前 H=8 作为机制验证；正式 selection 改为一次 `Hmax=32` rollout，保存 `H=1/8/16/32` raw cumulative outcomes，再从同一 rollout 派生四个数据视图。此阶段不读取 TEST。
6. **TRAIN-only policy selection**：按固定三 seeds（`20261003/20261004/20261005`）运行 threshold、nonlinear-threshold 和 generic-MLP controls，保持 equal supervision；完成 feature audit、same-score/different-state、reviewer controls、NLL/accuracy/Brier/ECE/risk-coverage 和 validation-only calibration。
7. **生成 canonical selection lock**：`outputs/research_final_v2/manifests/FINAL_SELECTION_LOCK.json` 必须绑定 canonical model-20000 digest、policy split digest、selection protocol digest、选定 horizon/architecture/threshold/MLP/calibration。TEST 在此之前保持 fail-closed。
8. **锁后 TEST**：只有 lock PASS 后，才在 `formal/test_official/` 重新生成 official TEST counterfactual；`formal/test/` 下的 pre-lock 诊断即使完整也不得复用。随后运行 baseline、Oracle、threshold、state-threshold、MLP、JEV 的同 checkpoint 正式 replay。
9. **正式评测与报告**：运行 TrackEval、cross-view CVIDF1/CVMA、Oracle gate、policy summary，写出 `FINAL_REPORT.json`；全部阶段完成后才写 `outputs/research_pipeline/PIPELINE_COMPLETE.json`。

## 4. 保守时间估计（从 2026-10-05 约 06:56 UTC 快照开始）

这些是资源和当前速度推导的区间，不是承诺；formal builder 没有可靠的中间百分比，只有终态 manifest 才算完成。

| 阶段 | 估计耗时 | 影响因素 |
|---|---:|---|
| native strict 剩余测试集 | 约 1–3 小时 | 当前 stream 约 85.3%，但按视频组落盘；每个视频组耗时差异很大，必须等 22 组全部完成并写出 COMPLETE manifest |
| traced strict + trajectory gate + 4A/4B replay gates | 约 2–5 小时 | traced 必须在 GPU0 顺序复跑全 VISION_test；4B 还要逐序列重放 frozen cache |
| 当前 TRAIN formal shards | 约 1–12 小时 | 只在完整 shard 结束时落盘；CPU、内存和 GMT association 开销不稳定 |
| TRAIN Hmax=32 + 派生 H=1/8/16/32 | 约 2–8 小时 | 只做一次最慢的 H=32 association rollout；派生视图为 CPU/IO 操作 |
| 三 seed selection、controls、audit、calibration | 约 2–8 小时 | 主要为 CPU 离线训练/评估，可与部分后处理重叠但不能读取 TEST |
| selection lock 后 TEST + Oracle/baseline/JEV + TrackEval | 约 2–8 小时 | 包括官方 replay、cross-view 和 Oracle gate |
| **完整关键路径** | **从当前快照保守按约 10–36 小时** | 以 PASS manifest 和 lock 为结束条件，不以进程存活时间为结束条件；formal builder 和不同视频组耗时是最大不确定项 |

当前最不确定的是 formal counterfactual builder；如果 shard 因 OOM/终止失败，保留 `.incomplete.*` 和日志，再按同一 shard 分片策略恢复，不能把部分 JSON 当作有效结果。

## 5. 产物与验收清单

最终必须存在并通过验证：

- `outputs/stage1_single_gpu/model_16000.pth`
- `outputs/stage2_single_gpu/model_20000.pth` 及独立 checkpoint validation PASS
- `off/same_gpu_off_gate.json`：strict same-GPU trajectory `PASS`（discrete exact + float tolerance）
- `off/trace_off_contract_gate.json`：4A `PASS`
- `off/v2_simulator_off_replay_gate.json`：4B `PASS`
- TRAIN formal merged manifests 的 provenance 全字段一致；不同 worktree 未通过 equivalence 前必须 `REFUSE_TO_MERGE`
- 一次 Hmax=32 TRAIN rollout 及其 H=1/8/16/32 派生 datasets
- `manifests/jev_selection_protocol_final.json`：TRAIN-only、三 seeds、digest 完整、`PASS`
- `manifests/FINAL_SELECTION_LOCK.json`：canonical model-20000、TEST 未参与 selection、`PASS`
- `official/` 下 baseline/Oracle/threshold/MLP/JEV 的 replay、TrackEval 和 cross-view 结果
- `outputs/research_final_v2/FINAL_REPORT.json`
- `outputs/research_pipeline/PIPELINE_COMPLETE.json`

在最终审计前，历史 cross-GPU OFF 数值只能作为 diagnostic/reporting evidence；pre-lock TEST 只能作为 diagnostic evidence；二者都不能授权 selection 或 official claim。

## 6. 安全操作规则

- 不杀当前 PID，不启动第二个 strict gate，不重复占用 GPU0。
- `formal/test/DO_NOT_USE_FOR_SELECTION` 下的 pre-lock TEST 只允许作为隔离诊断；完成后用 `reproduction_tools/quarantine_prelock_test.py` 生成 `prelock_quarantine_manifest.json`，其中必须含 `prelock=true`、`selection_authority=false`、`official_result_authority=false`，不得进入 selection 或 final report。
- 不在 final selection lock 前读取或生成新的 official TEST counterfactual；锁后只允许写入 `formal/test_official/` 并重新生成。
- 每个 shard manifest 必须锁定 `checkpoint_sha256/config_sha256/source_commit/counterfactual_engine_version/state_schema_version/utility_definition/association_backend/annotations_sha256/cache_sha256/trace_sha256/horizon` 及 authority 字段；任一不一致，merge 直接 `REFUSE_TO_MERGE`。
- 低内存 worktree 只有在相同小视频子集通过 record count exact、state exact、legal actions exact、utility tolerance、best action exact 的 builder equivalence 后才可混入正式 shard。
- 不删除 checkpoint、trace、cache、日志或 `.incomplete.*`；空间不足时先报告并做可恢复归档。
- 不把 `/data1` 下大体量数据、checkpoint、outputs 或用户 `results/` 提交 Git；Git 只提交源码、协议和本说明书。
- 当前旧 parent 完成或终止后，先核对所有子进程和 manifest，再启动最新 `run_final_v2_pipeline.py`；不得仅根据旧日志尾部的 FAILED 文本判断当前 live child 已失败。
