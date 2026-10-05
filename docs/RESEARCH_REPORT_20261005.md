# WWW / GMT-JEV 阶段研究报告与结果说明

更新时间：2026-10-05 21:25（Asia/Shanghai）。本报告是当前可审计进度快照；在最终 selection lock、official TEST、完整评测和 `PIPELINE_COMPLETE.json` 出现之前，不把阶段结果表述为最终 JEV 结论。

## 1. 结论摘要

目前已经完成并可复现的部分如下：

- Stage1/Stage2 checkpoint 已验证；当前 canonical Stage2 是 `model_20000.pth`，包含 optimizer state，iteration/scheduler/global iteration 均为 20000，finite/reload 校验通过。
- reviewer-proof v2 的协议、来源提交绑定、轨迹/trace/simulator gate、H=32 规划、三 seed 聚合和主机资源 gate 已写入代码与文档。
- strict same-GPU OFF 的 native 推理已在 GPU0 完成 22 个 group、58,038 张图，manifest 为 `COMPLETE`。
- strict same-GPU OFF 的 traced 推理正在同一 GPU0 顺序执行；native/traced 已使用互相隔离的 writer cache，避免重复写入旧 native cache。
- 现有 canonical Stage2 OFF baseline 已有 TrackEval/CV audit 数值，但这些数值只用于基线记录，不能替代 formal JEV 选择和 official TEST。

尚未完成的部分：traced OFF、同卡轨迹等价 gate、4A trace contract、4B simulator replay、TRAIN-only H=32 反事实数据、三 seed 控制器选择、最终锁后 official TEST、Oracle/baseline/JEV 正式评测。因此目前不能宣称“JEV 最终有效”或给出最终 GO/NO-GO。

## 2. 代码与环境基线

| 项目 | 当前值 |
| --- | --- |
| canonical repo | `/data1/liuyeqiang/WWW` |
| formal worktree | `/data1/liuyeqiang/WWW_jev_v2` |
| branch | `jev/reviewer-proof-v2` |
| latest local commit | `f3187b8` (`Isolate native and traced OFF caches`) |
| checkpoint | `/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth` |
| checkpoint SHA256 | `cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8` |
| config | `/data1/liuyeqiang/WWW_jev_v2/configs/VISION_test.yaml` |
| config SHA256 | `bdaeca71d875e8824c3eaf825967a7ba032514297642a6aabe7ad10a740be94a` |
| runtime | `/home/liuyeqiang/anaconda3/envs/GMT/bin/python` |

Stage1 checkpoint 为 `model_16000.pth`；其文件名中的 local iteration 不能单独解释 global lineage，最终以 checkpoint validation manifest 为准。数据、checkpoint、trace、cache、outputs 和用户已有 `results/` 均有意不提交 Git。

## 3. 已完成的验证与改动

### 3.1 Checkpoint 与基础 invariant

- Stage2 `model_20000.pth`：validation `PASS`。
- checkpoint reload、model finite、optimizer state：`PASS`。
- OFF evidence gate invariants：`PASS`。
- JEV perception-cache invariants：`PASS`。
- JEV selection gate invariants：`PASS`。
- GMT Python 的相关源码 `py_compile`：`PASS`。
- `git diff --check`：`PASS`。

### 3.2 reviewer-proof v2 代码方向

当前本地提交链已包含：

- OFF native/traced 的严格同卡执行和可审计 provenance；
- 4A trace contract 与 4B simulator replay 的 fail-closed gate；
- counterfactual shard 的 checkpoint/config/source/engine/state/utility/backend/annotation/cache/trace/horizon 绑定；
- Hmax=32 一次 rollout、H=1/8/16/32 派生视图的协议；
- 三 seed validation mean ± sample std 聚合，禁止挑单个最好 seed；
- selection 前 host RAM、swap、磁盘资源 gate；
- pre-lock TEST 隔离和最终 `FINAL_SELECTION_LOCK` 约束；
- 2026-10-05 修复：native 与 traced 不再共用 writer cache。修复前的失败 traced 运行已保留在 `traced.incomplete.20261005T124609Z/`，没有被当作有效结果。

## 4. 已产生的结果

### 4.1 Strict native OFF

输出目录：

`/data1/liuyeqiang/WWW/outputs/research_final_v2/off/same_gpu_strict/native`

- inference manifest：`status=COMPLETE`，return code `0`；
- 22 个视频 group 全部完成；
- 共处理 58,038 张图；
- prediction JSON：`inference_VISION_test/coco_instances_results.json`；
- prediction JSON SHA256：`2e94420f5160d33a50532e1b206318c4d6422ead4a259413da468a25031804ab`；
- native trace：`/data1/liuyeqiang/WWW/outputs/research_final_v2/off/same_gpu_strict/native_off.jsonl`。

这表示 native 推理本身完成，不表示 native 与 traced 的轨迹比较 gate 已通过；后者必须等待 traced 完整结束后再运行。

### 4.2 已记录的 canonical Stage2 OFF baseline

该结果来自已验证 Stage2 checkpoint 的既有 OFF evaluation，报告文件为 `docs/STAGE2_OFF_TRACKING_RESULTS.md`。它不参与 JEV policy selection。

| HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Frag |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8.3140 | 6.4506 | 11.9220 | 8.5297 | -70.2413 | 3,247 | 60,103 |

统计量：`Dets=527,474`、`GT_Dets=580,178`、`IDs=932`、`GT_IDs=685`。cross-view audit 的 sequential-camera convention 为 `CVIDF1=8.1854`、`CVMA=-70.2676`；interleaved convention 为 `CVIDF1=8.1854`、`CVMA=-70.9970`。由于下载 GT 存在重复行，TrackEval 使用了明确记录的 permissive duplicate-ID audit mode，未修改 GT。

这些数值只能作为当前 OFF baseline artifact 的说明，不能作为最终论文结论，也不能证明 JEV 相对 baseline 已经提升。

## 5. 当前运行与下一步

当前 strict traced OFF 使用同一个 checkpoint/config、同一可见 GPU0，输出为：

- output：`/data1/liuyeqiang/WWW/outputs/research_final_v2/off/same_gpu_strict/traced`；
- trace：`/data1/liuyeqiang/WWW/outputs/research_final_v2/off/same_gpu_strict/traced_off.jsonl`；
- 独立 cache：`/data1/liuyeqiang/WWW/outputs/research_final_v2/off/same_gpu_strict/cache_traced/index.jsonl`。

当前 traced manifest 仍为 `RUNNING`。native 完成后 traced 已正常进入后续 group，未再出现旧的 `refusing to overwrite cached perception record` 错误。不得因为单个 group 暂时没有新增日志而终止它。

traced 完成后的强制顺序是：

1. same-GPU OFF trajectory gate；
2. 4A `TRACE_OFF_CONTRACT_GATE`；
3. 4B `V2_SIMULATOR_OFF_REPLAY_GATE`；
4. TRAIN-only formal counterfactual 与 Hmax=32 派生数据；
5. 三 seed controls、feature audit、calibration 和 validation-only selection；
6. 生成并校验 `FINAL_SELECTION_LOCK.json`；
7. 锁后重新生成 official TEST，运行 baseline、Oracle、threshold、MLP、JEV 和 TrackEval/cross-view；
8. 只有所有 manifest/gate/report 完成后，才写入 `FINAL_REPORT.json` 与 `PIPELINE_COMPLETE.json`。

## 6. GitHub 推送状态

本地 branch 和提交已准备好，当前 worktree 仅有用户原有的未跟踪 `results/`，没有把它加入提交。已尝试：

```bash
cd /data1/liuyeqiang/WWW_jev_v2
git push origin jev/reviewer-proof-v2
```

但当前环境没有 GitHub HTTPS username/token 或 SSH 凭据，返回：

`fatal: could not read Username for 'https://github.com': No such device or address`

因此目前不能诚实地声称已经上传到 GitHub。认证配置完成后，直接执行同一条命令即可推送当前分支；推送目标为 `https://github.com/LYQ1107/WWW.git`。本地源码备份还包括 `/data1/liuyeqiang/WWW_jev_v2_20261005.bundle`。

## 7. 最终结果判定标准

最终研究结果必须同时满足：所有 formal shard provenance 一致；OFF 4A/4B gate 为 `PASS`；selection 只读取 TRAIN/validation 且三 seed 聚合；`FINAL_SELECTION_LOCK` 绑定 canonical model/config/digest；official TEST 只在 lock 后生成；Oracle、baseline、JEV 的评测可复现；`FINAL_REPORT.json` 与 `PIPELINE_COMPLETE.json` 均通过校验。当前尚未满足这些条件，所以本报告明确区分“已完成 artifact”和“待完成 formal result”。
