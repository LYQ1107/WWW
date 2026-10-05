# WWW / GMT-JEV 阶段研究报告与结果说明

更新时间：2026-10-05 21:11 UTC。本报告是当前可审计进度快照；在三方法第一轮验证、gate 和正式 tracking 评测完成之前，不把阶段结果表述为最终 JEV 结论。

## 1. 结论摘要

目前已经完成并可复现的部分如下：

- Stage1/Stage2 checkpoint 已验证；当前 canonical Stage2 是 `model_20000.pth`，包含 optimizer state，iteration/scheduler/global iteration 均为 20000，finite/reload 校验通过。
- Frozen canonical GMT baseline 的 TrackEval/CV 数值已经锁定，本轮不重新训练 GMT、不重新执行完整 baseline inference，也不重复生成 baseline trace/cache/predictions。
- 全量 TRAIN H=8 正式反事实数据已经按 24 个视频分片启动；每个视频由一个独立 worker 顺序生成，使用同一个 canonical Stage2 checkpoint、cache、trace 和 H=8 协议。
- `finalize_jev_full_h8.py` 已加入并推送，用于在 24 个分片全部 COMPLETE 后做 provenance、记录数、schema、hash 和重复事件审计，再原子合并。
- 第一轮三方法脚本已锁定为 `seed=20261003`：Learnable Threshold、Generic MLP、Full JEV；训练超参数、样本权重、targets 和 shared H=8 数据保持一致。

尚未完成的部分：24 个 H=8 分片、正式合并与审计、compact dataset/policy split、三方法单 seed 训练与 validation 指标、JEV gate，以及 gate 通过后的三方案 tracking comparison。因此目前不能宣称“JEV 最终有效”或给出最终 GO/NO-GO。

## 2. 代码与环境基线

| 项目 | 当前值 |
| --- | --- |
| canonical repo | `/data1/liuyeqiang/WWW` |
| formal worktree | `/data1/liuyeqiang/WWW_jev_v2` |
| branch | `jev/reviewer-proof-v2` |
| latest local commit | `6ba4424` (`Add full H8 per-video shard finalizer`) |
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
- OFF evidence gate invariants：`PASS`；冻结 baseline 本轮直接复用。
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
- 第一轮单 seed validation 聚合（`20261003`）；三 seed mean/std、ensemble 和 robustness 暂时延期；
- selection 前 host RAM、swap、磁盘资源 gate；
- pre-lock TEST 隔离和最终 `FINAL_SELECTION_LOCK` 约束；
- 2026-10-05 新增：全量 H=8 per-video shard finalizer，拒绝缺片、采样/截断、provenance 不一致、记录数不一致和重复事件。

## 4. 已产生的结果

### 4.1 Frozen canonical GMT baseline

本轮固定复用以下同一个 canonical `model_20000.pth` 的结果，不重新生成 baseline：

| HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Frag | CVIDF1 | CVMA |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 67.442 | 66.278 | 68.992 | 82.239 | 80.942 | 3,092 | 8,004 | 79.0248 | 80.9276 |

这些数字只作为最终表格的固定第 0 行；新增三种方法必须使用相同数据、checkpoint 和评测协议。

### 4.2 Full TRAIN H=8 formal build

运行根目录：`/home/liuyeqiang/WWW_jev_full_h8_runtime`。

- partition manifest：`status=COMPLETE`，24 个视频，预期 `1,112,173` 条 decision。
- 当前快照：`RUNNING=24`、`PENDING=0`、`COMPLETE=0`、`FAILED=0`；累计写入约 `27,130` 条（约 `2.44%`）。
- 当前 24 个临时分片仍在更新；GPU 0/4/6/8/9 均接近满载。当前短窗速率约 `3.4 decision/s`，剩余时间按负载估计约 65–90 小时，随视频和机器负载变化。
- 正式 `video_XX/manifest.json` 尚未出现；因此现在禁止运行 finalizer、compact、policy split 或训练。

当前只清理/复用正式协议允许的运行产物，不触碰用户已有 `results/`、checkpoint、cache 和其他任务。

### 4.3 第一轮固定协议

- seed：仅 `20261003`；暂不计算 mean ± std、三 seed ensemble、seed robustness 或 best-seed comparison。
- 方法：Learnable Threshold、Generic MLP、Full JEV。
- 指标：Val NLL、Best-action Accuracy、Brier、ECE、Validation Utility。
- 只有当 Full JEV 的 Validation Utility 同时高于两个 control，才进入三方法完整 tracking comparison。

## 5. 当前运行与下一步

当前正在运行的是全量 H=8 formal shard build；正式分片完成后的强制顺序是：

1. 等待并审计 24 个 `video_XX/manifest.json`；
2. 运行 `finalize_jev_full_h8.py`，得到单一完整 JSONL 及 PASS manifest；
3. 生成 H=8 compact dataset 和固定 sequence-disjoint policy split；
4. 在相同 full TRAIN 数据上各训练一次三种方法，并做 validation-only calibration/aggregate；
5. 若 Full JEV 同时胜过两个 control，才运行冻结 baseline、Oracle、三方法的正式 tracking comparison；
6. 记录所有 hash、gate、指标和可复现命令，形成最终报告。

## 6. GitHub 推送状态

当前分支已经成功推送到 GitHub，远端已核验：

- repository：<https://github.com/LYQ1107/WWW>
- branch：<https://github.com/LYQ1107/WWW/tree/jev/reviewer-proof-v2>
- 阶段报告：<https://github.com/LYQ1107/WWW/blob/jev/reviewer-proof-v2/docs/RESEARCH_REPORT_20261005.md>
- 运行说明书：<https://github.com/LYQ1107/WWW/blob/jev/reviewer-proof-v2/docs/RESEARCH_RUNBOOK.md>
- 已核验远端 commit：`6ba44245f76f8a8fdd482049084e29a3ac345101`

第一次 HTTPS 推送因缺少 username/token 失败；随后使用专用 SSH 公钥认证，并通过 GitHub SSH-over-443 和当前代理完成推送。worktree 的 SSH 配置已保留，后续提交可以继续推送。用户原有的未跟踪 `results/`、数据、checkpoint、trace、cache 和 outputs 没有加入 Git。本地源码备份还包括 `/data1/liuyeqiang/WWW_jev_v2_20261005.bundle`。

## 7. 最终结果判定标准

最终研究结果必须同时满足：24 个 formal shard provenance/记录数/schema/hash 一致；compact dataset 与 sequence split 可复现；三种方法均在固定 `seed=20261003` 完成训练、校准和第一轮 validation 指标；若 gate 通过，Oracle、baseline、三方法 tracking/cross-view 评测可复现；最终报告明确记录 gate 通过或失败及其证据。当前尚未满足这些条件，所以本报告明确区分“已完成 artifact”和“待完成 formal result”。
