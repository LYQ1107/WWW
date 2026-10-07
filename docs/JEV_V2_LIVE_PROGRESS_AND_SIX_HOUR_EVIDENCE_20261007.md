# Video01 corrected v2 live progress and six-hour evidence

Snapshot: **2026-10-07 10:57:46 UTC**  
Classification: **live operational evidence; not a final experiment result**

## Direct conclusion

The formal corrected video01 v2 builder is still running. The same OS process
(`PID 12163`) started at **2026-10-07 04:15:18 UTC** and, at the snapshot,
had been alive for **06:42:28** with **06:42:33** CPU time, `Rsl` state, 100%
CPU, and 3,165,152 KiB RSS. This is concrete evidence that the job has run for
more than six hours under one PID. It is not evidence that the final dataset is
complete.

The formal output directory still has no `video01_records.jsonl` and no
`video01_records.jsonl.manifest.json`. Therefore no final v2 acceptance, controller selection, or
three-way tracking metrics are claimed in this report.

## Frozen provenance

The builder is running from the frozen source commit:

```text
worktree: /data1/liuyeqiang/WWW_rng_fix_v4
branch:   jev/counterfactual-rng-isolation-v4-20261006
commit:   4108f18f5040432f68d55872e81e9f76e9acd08f
```

Its inputs are the existing video01 OFF trace, cached perception, the GMT
`model_20000.pth`, VISION_test configuration, and the `formal_gmt_transformer`
backend. The complete argv and input hashes are recorded in
`reports/JEV_RNG_V4/VIDEO01_V2_LIVE_EVIDENCE_20261007.json`.

## Six-hour process evidence

| Field | Observed value |
|---|---|
| PID | `12163` |
| Start | `2026-10-07 04:15:18 UTC` |
| Snapshot | `2026-10-07 10:57:46 UTC` |
| OS elapsed time | `06:42:28` |
| CPU time | `06:42:33` |
| State | `Rsl` |
| CPU | `100%` |
| RSS | `3,165,152 KiB` |
| GPU observation | GPU 0, process memory 1,231 MiB in the 10:57 UTC `nvidia-smi` snapshot |

The exact evidence is machine-readable in
[`VIDEO01_V2_LIVE_EVIDENCE_20261007.json`](../reports/JEV_RNG_V4/VIDEO01_V2_LIVE_EVIDENCE_20261007.json).

## Other jobs at the snapshot

- PID `5457`: bounded cost diagnostic (`max-events=2400`), still running on GPU 4;
  at the same snapshot it had run for `00:38:08` and had not written its final
  report yet.
- PID `9824`: targeted frames 210–220 reactivation diagnostic completed. It
  emitted 113 diagnostic records, including 5 reactivation records, with zero
  skipped events in 222.136 seconds. This is a diagnostic artifact only.
- PIDs `2843/2849`: aftercare waiter is still waiting for the formal v2
  manifest; provenance, parity, stability, and closed-loop stages have not
  started.

No active job was stopped, migrated, or restarted for this snapshot.

## Current results and interpretation

### 1. Recomputation scope audit

`RECOMPUTATION_SCOPE_AUDIT_VIDEO01_20261007.json` is diagnostic-only. On a
20-record same-input comparison, state projections matched exactly (20/20),
but reward/label projections did not (0/20; 20 mismatches). A repeat of the
current sample was byte-identical, while a historical 60675 replay reproduced
the current sample rather than the old artifact. The old reward, horizon,
target-probability, best-action, and sample-weight fields therefore cannot be
reused. Full current-v2 recomputation remains required.

### 2. Candidate parity

`CANDIDATE_PARITY_DIAGNOSTIC_VIDEO01_20261007.json` is **BLOCKED** for the
stale replay: candidate IDs/order and legacy OFF actions match, but candidate
scores and selected proposal IDs do not. The capped accounting is 225
mismatches, 100 native-only events, 100 replay-only events, 25 candidate-score
errors, and 8 chosen-proposal mismatches. Fresh parity must be run against the
completed v2-bound records.

### 3. Cost and mismatch-window diagnostics

The completed 20-event cost sample took 21.962 seconds. It measured 857
proposal calls (7.893 s), 908 association steps (7.386 s), 51 branch clones,
and 862 reactivation-context calls. Proposal plus step consumed about 69.6%
of wall time in that sample.

The targeted native mismatch window (frames 210–220) completed separately with
113 records: 57 MATCH, 51 MEMORY, and 5 REACTIVATION decisions. It is useful
for investigating candidate parity, but it is not a replacement for the formal
v2 artifact.

### 4. Controller status

The existing corrected small-H8 Threshold, Generic MLP, and JEV checkpoints
pass a loader/wrapper smoke test for seed `20261003`. That smoke test does not
establish video01 runtime parity or tracking improvement, so it is not reported
as a final comparison.

## What is not available yet

There are currently no defensible final v2 values for Val NLL, accuracy, Brier,
ECE, utility, HOTA, AssA, IDF1, MOTA, IDSW, or Frag. The final three-way
closed-loop stage must wait for the formal v2 manifest and its hard gates.

Full H8 remains explicitly paused and noncanonical because of the previous
source-commit mixing risk and premature authorization. It must not be resumed
from the mutable worktree before the corrected video01 gates pass and a new
frozen worktree is created.

## Files pushed for review

This report and its JSON evidence are published on the audit branch:

```text
https://github.com/LYQ1107/WWW/tree/jev/audit-diagnostics-20261007
```

The branch contains the earlier recomputation-scope, candidate-parity, cost,
checkpoint-wrapper, and recovery-protocol artifacts as well. The canonical
working branch remains at `4108f18f...`; this publication does not modify that
frozen commit or the running builder.
