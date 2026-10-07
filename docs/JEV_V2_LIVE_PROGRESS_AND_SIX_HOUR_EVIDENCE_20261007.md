# Video01 corrected v2 live progress and six-hour evidence

Snapshot: **2026-10-07 12:05:14 UTC**  
Classification: **completed-run evidence plus live small-gate status; not a final experiment result**

## Direct conclusion

The formal corrected video01 v2 builder ran under one OS process (`PID 12163`)
from **2026-10-07 04:15:18 UTC** to **2026-10-07 11:10:35 UTC**, for
**06:55:17.54**, and durably wrote **8995/8995** records. This is the requested
six-hour evidence. It is not, by itself, a semantic acceptance result.

The subsequent aftercare completed at **2026-10-07 11:18:13 UTC** and returned
`NO_GO_RUNTIME_SEMANTICS_GATE`; no final v2 controller selection or three-way
tracking metrics are claimed.

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
| Completion | `2026-10-07 11:10:35 UTC` |
| Total elapsed time | `06:55:17.54` |
| Records | `8995/8995` |
| JSONL SHA-256 | `1da2ed9e7eec43c7cf139754d873f78a299c344ff74d65c6866b92f18f3da79` |
| Manifest | `PASS` |
| Aftercare decision | `NO_GO_RUNTIME_SEMANTICS_GATE` |

The exact evidence is machine-readable in
[`VIDEO01_V2_LIVE_EVIDENCE_20261007.json`](../reports/JEV_RNG_V4/VIDEO01_V2_LIVE_EVIDENCE_20261007.json).

## Other jobs at the snapshot

- PID `5457`: bounded cost diagnostic (`max-events=2400`), still running on GPU 4;
  at the same snapshot it had run for `00:38:08` and had not written its final
  report yet.
- PID `9824`: targeted frames 210–220 reactivation diagnostic completed. It
  emitted 113 diagnostic records, including 5 reactivation records, with zero
  skipped events in 222.136 seconds. This is a diagnostic artifact only.
- Aftercare has now completed. Provenance passed, but runtime feature parity,
  reactivation candidate parity, and three-repeat stability failed; the exact
  counts are in `VIDEO01_SIX_HOUR_RUNTIME_EVIDENCE_20261007.json`.

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
