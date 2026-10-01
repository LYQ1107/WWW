# Follow-up status

**Status:** COMPLETE — replay fidelity and snapshot gates PASS; isolated C1b/C2b audit completed with `STOP_NO_STRONG_CAUSAL_GATE`.

**Workspace:** `/data3/liuyeqiang/JEV_GMT`  
**Branch:** `jev-gmt-causal-prototype`  
**HEAD at start:** `416d543b24b7f04d06add750ced95f4a24c9f1f4`

## Preservation check

- First-round C1/C2 manifests and result CSVs are present and unchanged.
- No GMT/JEV training process is running.
- No formal Stage1/Stage2 training will be started.

## Gates

| Gate | Status | Evidence |
|---|---|---|
| Replay fidelity | PASS | `causal_followup/REPLAY_FIDELITY.json`; exact pre-filter trace equality for all three scenes and all 7,130 cached observations. The optional original post-filter dump was not collected in the first dump run. |
| Snapshot round-trip | PASS | `causal_followup/SNAPSHOT_ROUNDTRIP.json`; all three scenes have exact 10-frame continuation and RNG round-trip. |
| C1b/C2b causal gate | STOP — no strong state-causal GO | `causal_followup/results/FOLLOWUP_STATISTICS.json`; `causal_followup/STATE_CAUSAL_EVIDENCE_MATRIX.md` |

## Phase log

- Phase 0: preservation/read-only audit complete.
- Phase 1: event-isolated replay design is being implemented.
- Phase 2: observation cache and exact baseline replay completed; `causal_followup/REPLAY_FIDELITY.json` is PASS.
- Phase 3: serializable tracker state round-trip completed; `causal_followup/SNAPSHOT_ROUNDTRIP.json` is PASS.
- Phase 4–7: all 71 C1b and 256 C2b frozen events replayed from isolated snapshots; duplicate frame/view events were separately branched.
- Phase 8: C2b +5 paired Δ = +0.074 percentage points, 95% CI [-0.011, +0.139] percentage points; C1b-B/C did not produce a significant negative repair effect at all required horizons.
- Final gate: `STOP_NO_STRONG_CAUSAL_GATE`; no JEV-GMT source/design/prototype work was started.

Last updated: 2026-10-01.
