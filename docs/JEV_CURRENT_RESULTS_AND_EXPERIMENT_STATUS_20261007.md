# JEV 当前运行结果与实验状态（2026-10-07）

本页是当前可审查的状态快照。GitHub 中保存的是报告、时间戳、源代码 commit 和 SHA-256 证据；原始 JSONL、模型和缓存仍保留在本机运行目录，没有把大文件伪装成论文结果。

## 结论先说

- corrected formal video01 H=8 v2 已真实运行约 **6 小时 55 分 17 秒**，完成并落盘 **8995/8995** 条记录。
- 文件和 provenance 完整性通过，但 runtime semantics gate 没通过：runtime feature parity、reactivation candidate parity、stability 均 FAIL。因此这批记录**不能作为最终训练数据或论文结果**。
- 当前 segmented small gate 仍在运行，Full H8 保持暂停、非 canonical、未授权。
- 当前 3 workers/GPU 重启后的 queue 审计没有发现重复分片、重复 active chunk 或 failure file；短时 supervisor 计数约 4.27 records/s，但暂不宣称 3 倍加速。

## 1. video1 六小时证据

证据报告：[`VIDEO01_SIX_HOUR_RUNTIME_EVIDENCE_20261007.json`](../reports/JEV_RNG_V4/VIDEO01_SIX_HOUR_RUNTIME_EVIDENCE_20261007.json)

| 项目 | 证据 |
|---|---|
| Builder PID | 12163 |
| 开始 | 2026-10-07 04:15:18 UTC |
| 完成 | 2026-10-07 11:10:35 UTC |
| 用时 | 06:55:17.54 |
| records | 8995 |
| JSONL 行数 | 8995 |
| records SHA-256 | `1da2ed9e7eec43c7cf139754d873f78a299c344ff74d65c6866b92f18f3da79` |
| source commit | `4108f18f5040432f68d55872e81e9f76e9acd08f` |
| checkpoint SHA-256 | `cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8` |

问题不在“有没有跑完”：它跑完了。问题在于后验语义门禁：

- provenance：PASS；checkpoint wrapper：PASS；
- runtime feature parity：FAIL，8995 条比较中最大绝对误差 `0.006004035472869873`，容差 `2e-5`；
- reactivation candidate parity：FAIL，225 个 mismatch；candidate ID/order 对，但 candidate scores 和 chosen proposal 不完全一致；
- runtime feature stability：FAIL，3 次重复均出现误差；
- closed-loop：因 pre-gate failure 未运行；
- 决策：`NO_GO_RUNTIME_SEMANTICS_GATE`，`FULL_H8_AUTHORIZED=false`。

因此不能因为“超过六小时并且 8995 条都写出来”就把它升级成 canonical 或论文结果。

## 2. 当前 segmented small gate

并发审计：[`SEGMENTED_CONCURRENCY_CHANGE_AUDIT_20261007.json`](../reports/JEV_RNG_V4/SEGMENTED_CONCURRENCY_CHANGE_AUDIT_20261007.json)

当前 runtime：`/home/liuyeqiang/WWW_jev_rng_v4_runtime/segmented_gate_20261007_v4`

- source commit 固定为 `1d2711e80ac5fa00806fd9eed30e90cd51df6b30`；
- 旧 driver PID 15541 已 graceful stop，新 driver PID 13608；
- 7 张卡 `[2,3,5,6,7,8,9]`，当前 21 个 worker，即 3 workers/GPU；
- 7 个已完成 chunk（1406 条 finalized records）复用；旧的 850 条 partial 从 snapshot 重算；7 份 interrupted partial 文件保留作证据；
- 观察时 queue：`60 PENDING / 21 RUNNING / 7 COMPLETE`，总计划 17491 条；root supervisor counter 为 2250（包含 running partials）；
- video01：`23 pending / 15 running / 7 complete`；video06：`20 / 6 / 0`；video07：`17 / 0 / 0`；
- active chunk 名称无重复、每个 video 的 record ranges 唯一、没有 failure files；
- 11:42:10–11:45:10 supervisor counter 从 1481 到 2250，观测约 4.27 records/s。这是运行性证据，不是稳定 ETA，也没有拿它宣称 3 倍加速。

当前 21 个 worker 的 GPU 2/3/5/6/7/8/9 利用率约 94–100%，每卡约 3683 MiB / 32508 MiB。作业继续保留，不做 kill -9、不迁移 video1。

## 3. 已通过的有限验证

- early bounded gate（frames 210–216）：70/70 records exact，candidate mismatch 0；
- late bounded gate（frames 881–890）：102/102 records exact，167 个 candidate events 比较，mismatch 0；
- basic tests：5/5 PASS；
- 这些是 segmented/chunk contract 的 bounded evidence，不等价于 full video1 runtime gate PASS。

## 4. 尚未完成与下一步

1. 保留当前 small gate，完成 video01/video06/video07 的 records。
2. 对完整 small gate 做 per-video provenance、runtime parity、candidate parity、stability。
3. 只有 gate 通过后，才运行 video1 held-out closed-loop；当前不重新训练已经完成的 video6/video7 三模型。
4. 只有 corrected video1 closed-loop 得到 GO，才重新冻结唯一 source commit 并授权 Full H8；当前 Full H8 仍是 `PAUSED_NONCANONICAL`。

相关已推送分支：

- [audit-diagnostics-20261007](https://github.com/LYQ1107/WWW/tree/jev/audit-diagnostics-20261007)
- [video01-reactivation-box-parity-fix-20261007](https://github.com/LYQ1107/WWW/tree/jev/video01-reactivation-box-parity-fix-20261007)
