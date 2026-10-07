# JEV RNG v4 current progress — 2026-10-07

**Snapshot:** `2026-10-07T12:05:14Z`
**Classification:** live research status, not a final paper result

## 结论先行

当前 Full H8 **已经暂停，且不再被视为 canonical build**。原因有两个：

1. 它在 corrected video01 的完整 runtime gate 通过前就开始了；
2. 它在持续变化的开发 worktree 上运行，期间发生了影响 counterfactual semantics 的 reactivation 修改，因此 original、slot2、slot3 worker 不能被证明来自同一个 source commit。

当前唯一主线是：

```text
corrected video01 完成
→ provenance
→ MATCH / MEMORY / REACTIVATION runtime parity
→ reactivation candidate parity
→ numerical stability
→ GMT OFF / Threshold / MLP / JEV closed-loop
→ GO / NO-GO
→ 只有 GO 才允许重新授权 Full H8
```

权威机器可读状态：
`reports/JEV_RNG_V4/CURRENT_GO_NO_GO_20261007.json`

暂停证据：
`reports/JEV_RNG_V4/PRE_VIDEO01_GATE_SPECULATIVE_H8_PAUSE_20261007.json`

## 当前运行状态

### segmented small gate（当前仍在运行）

- MPS 恢复后的 driver PID `1819` 存活，7 张卡 `[2,3,5,6,7,8,9]`，28 workers（4/GPU）。
- 最新快照：`13734/17491` 条（该计数包含 running partials），`55 COMPLETE / 28 RUNNING / 5 PENDING`。
- video01：`32 complete / 11 running / 2 pending`；video06：`16 / 9 / 1`；video07：`7 / 8 / 2`。
- MPS 前后 bounded equivalence：早期 `70/70`、晚期 `102/102` exact，candidate events（晚期）`167`，mismatch `0`。
- 这是 small-gate live evidence，不是 Full H8 授权，也不是最终论文结果。机器可读报告：`reports/JEV_RNG_V4/SEGMENTED_MPS_SWITCH_AND_LATE_EQUIVALENCE_20261007.json`。

### segmented small gate closed-loop 已完成

- video01 的四方案 closed-loop 已完成；JEV 相对 GMT OFF 为 `ΔHOTA +2.402`、`ΔAssA +4.873`、`ΔIDF1 +3.250`、`ΔIDSW -272`。
- Threshold/MLP 出现严重 tracking collapse，已经记录为 WARNING；完整汇总见 `reports/JEV_RNG_V4/SEGMENTED_SMALL_GATE_CLOSED_LOOP_VIDEO01_20261007.json`。
- 该结果只支持 JEV pilot continuation，不直接授权 Full H8；当前仍需完成 canonical source freeze、完整 chunk-equivalence authorization 和 controller warning 处置。

### corrected video01 v2 已完成，但 aftercare 为 NO-GO

- builder PID `12163` 已于 `2026-10-07T11:10:35Z` 完成，生成 `8995` 条记录和 PASS manifest；正式 artifact provenance PASS。
- aftercare 于 `2026-10-07T11:18:13Z` 完成，但 runtime feature parity、reactivation candidate parity 和三次 stability 均 FAIL。
- corrected Threshold、Generic MLP、Full JEV 闭环没有运行；当前不得据此报告 tracking 指标。

### 已优雅暂停的 Full H8

Full H8 队列：
`/home/liuyeqiang/WWW_jev_rng_v4_runtime/full_h8_current_head/queue_state.json`

暂停后的队列状态：

| 状态 | 数量 | 说明 |
|---|---:|---|
| `PAUSED` | 12 | 原来正在运行但收到 `SIGTERM` 的 video shard |
| `PENDING` | 12 | 尚未启动的 video shard |
| `COMPLETE` | 0 | 没有一个 Full H8 shard 被当作正式完成 |
| `FAILED` | 0 | 没有用失败状态覆盖现场 |

本次暂停只使用 graceful `SIGTERM`，没有使用 `kill -9`。保留内容包括：

- 12 个 `.tmp` records 文件，共约 `21,217` 个完整 JSONL 行；
- queue、scheduler/worker logs、resource manifest、PID snapshot；
- 所有现场数据只能用于 speed profiling、debug 和 commit-mixing 调查；
- 禁止 final merge、policy training、paper metrics 和 canonical tracking evaluation。

Full H8 已统一标记为：

```text
PRE_VIDEO01_GATE_SPECULATIVE_BUILD
NOT_CANONICAL
MIXED_SOURCE_COMMIT_RISK
```

## 为什么不能把这次 Full H8 当 canonical

Full H8 名义上于 `2026-10-06T22:52:40Z` 启动。启动后代码发生了以下语义相关变化：

| 变化 | commit | 时间 |
|---|---|---|
| branch-local reactivation semantics | `03f1dfd98f23ae4baab53e9b994bbdcbb73122a6` | `2026-10-06T23:23:01Z` |
| video1 reactivation branch fix 文档/状态推进 | `68bba4b33deeaf33b9dfc6f8e99556e4bc8725e7` | `2026-10-06T23:33:26Z` |

现场 worker 使用的是 mutable worktree：
`/data1/liuyeqiang/WWW_rng_fix_v4`。

启动前可追溯到的 HEAD 证据为：

| 进程组 | 启动前观察到的 HEAD | worker manifest 是否记录实际 source commit |
|---|---|---|
| original scheduler/workers | `7594b1d2ce8e44be3f7f2ef80b65ceae87d79c0e` | 否 |
| slot2 | `a942dd8d6a0ff2be1e779f86924ef09a16e2cb54` | 否 |
| slot3 | `e9226035e8c55bc1c98c364048557c5ea4e89f9b` | 否 |

由于 worker 未记录真实启动 commit，不能证明所有记录由同一 source 生成。结论固定为：
`CURRENT_FULL_H8_CANONICAL = FALSE`。

## 已经完成、但不能越权替代 video01 gate 的工作

- GMT Stage1/Stage2 checkpoint 已存在并通过审计。
- 冻结 GMT OFF baseline：HOTA `67.442`、DetA `66.278`、AssA `68.992`、IDF1 `82.239`、MOTA `80.942`、IDSW `3092`、Frag `8004`。
- video06/video07 corrected provenance、runtime feature parity 和当前 formal chunk equivalence gate 已通过。
- corrected video06/video07 的 Threshold、Generic MLP、Full JEV 单 seed 三模型训练已完成；报告为
  `reports/JEV_RNG_V4/CORRECTED_V4_SMALL_H8_THREE_WAY_VIDEO06_VIDEO07.json`。
- 该三模型结果仍是 screening evidence，不能当作 video01 closed-loop 结果；不重新训练 video06/video07。

## 当前明确未完成

`VIDEO01_V2_HARD_GATES_20261007.json` 当前为 `PENDING`，并且 `FULL_H8_AUTHORIZED=false`。当前需要先修复并重新生成 v2 artifact，再按顺序完成：

1. 修复 stale-bank native geometry/runtime mismatch；
2. 重新生成 corrected video01 v2 `8995` 条记录；
3. provenance 和三种 question type 的 coverage；
4. runtime feature parity；
5. reactivation candidate parity；
6. numerical stability；
7. 用现有 corrected Threshold / MLP / JEV checkpoint 做 video01 held-out closed-loop；
8. 根据 tracking 结果做 GO / NO-GO。

只有 GO 后才做 Full H8 重启。此之前，不能用一百万条临时记录掩盖 small-gate 问题。

## Full H8 重启前的硬约束

必须先确定唯一：

```ini
CANONICAL_H8_COMMIT=<exact 40-character SHA>
```

然后创建：

```text
/data1/liuyeqiang/WWW_h8_frozen_<CANONICAL_H8_COMMIT>
```

新的 scheduler、worker、slot2、slot3、chunk worker 和 aftercare 都必须从这个固定 worktree 启动。代码已经加入以下 fail-closed 规则：

- scheduler 启动前和每次 launch 前检查 `git rev-parse HEAD`；
- worker 启动时检查 worktree 路径和 exact commit；
- queue 绑定 `canonical_h8_commit` 与 `source_worktree`；
- 每个 worker manifest 记录 `source_commit`、`transformer_sha256`、`counterfactual_engine_sha256`、`adapter_sha256`；
- resource manifest 同样记录这些字段；
- 缺少任一 video01 hard gate 时，Full H8 authorization 直接失败。

## 状态文件

- `reports/JEV_RNG_V4/CURRENT_GO_NO_GO_20261007.json`
- `reports/JEV_RNG_V4/PRE_VIDEO01_GATE_SPECULATIVE_H8_PAUSE_20261007.json`
- `reports/JEV_RNG_V4/VIDEO01_CORRECTED_HARD_GATES.json`
- `reports/JEV_RNG_V4/VIDEO01_V2_AFTERCARE_GATE_RESULT_20261007.json`
- `docs/JEV_RNG_V4_LIVE_PROGRESS_20261007.md`

当前没有正向 JEV tracking claim，也没有授权 canonical Full H8 的结论。
