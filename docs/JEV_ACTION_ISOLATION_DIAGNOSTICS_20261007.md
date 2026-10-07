# JEV action-isolation diagnostics — 2026-10-07

These are held-out `video01 / 00002garden` screening diagnostics on the corrected segmented H=8 gate. They are not paper results and do not authorize the Full H8 build.

## Common binding

- Data source commit: `1d2711e80ac5fa00806fd9eed30e90cd51df6b30`
- Diagnostic runner commit: `84b6886ba38e2a816714c921004c83698f94693b`
- Records: 8,995; SHA256 `a174399cb979d5d1f9dc41366a4f07f6299c159c68e10e19222fedc656db7611`
- Native trace SHA256: `02773f217184600ce2eba7954e4a8fcd28c9070d8260e2a52ee07d670b0545b3`
- GMT checkpoint SHA256: `cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8`
- Runtime feature parity: 8,995/8,995, max absolute error `6.1035e-05`, zero OFF-action mismatches
- Reactivation candidate coverage: 174/174 evaluated, 0 wrong commits, 0 false reactivations

The GMT OFF pilot baseline is HOTA `86.5230`, AssA `85.1492`, IDF1 `94.6425`, MOTA `89.5791`, IDSW `282`.

## Isolation A: only learned reactivation

Mode: `baseline_memory_match`. MATCH and MEMORY are forced to GMT OFF; only REACTIVATION is learned.

All three controllers made the same runtime decisions in this mode: 48 `REACTIVATE_OLD`, 223 `START_NEW`, 0 `REASSOCIATE`, 4,301 `WRITE_MEMORY`; memory contamination was 0.

| Method | HOTA | AssA | IDF1 | MOTA | IDSW | ΔAssA | ΔHOTA |
|---|---:|---:|---:|---:|---:|---:|---:|
| GMT OFF | 86.5230 | 85.1492 | 94.6425 | 89.5791 | 282 | 0 | 0 |
| Threshold / MLP / JEV | 86.2159 | 84.5437 | 94.6425 | 89.5791 | 282 | -0.6055 | -0.3071 |

Reactivation alone is therefore a small but reproducible negative association perturbation. It is not the only failure source, but it cannot be treated as harmless.

## Isolation B: only learned matching

Mode: `baseline_memory_reactivation`. MEMORY and REACTIVATION are forced to GMT OFF; only MATCH is learned.

| Method | HOTA | AssA | IDF1 | MOTA | IDSW | ΔAssA | ΔHOTA |
|---|---:|---:|---:|---:|---:|---:|---:|
| GMT OFF | 86.5230 | 85.1492 | 94.6425 | 89.5791 | 282 | 0 | 0 |
| Threshold | 80.9985 | 74.6832 | 86.9312 | 80.3185 | 689 | -10.4660 | -5.5246 |
| MLP | 58.8311 | 39.4234 | 58.0811 | 72.5597 | 1,030 | -45.7258 | -27.6919 |
| JEV | 86.6731 | 85.5866 | 94.5752 | 95.5859 | 18 | +0.4374 | +0.1501 |

JEV has a positive matching-only pilot signal, but the result is still screening-only. Threshold and MLP are unsafe even after memory/reactivation are held fixed.

## Decision

The diagnostics explain why the unrestricted current-head three-way run was a quality `NO_GO`: a useful matching signal exists, but the complete policy changes mutable state through unsafe memory/reactivation decisions. The next research step is a controlled policy design/ablation around match-only or candidate-conditioned association, followed by the same provenance, parity, stability, and held-out closed-loop gates.

`FULL_H8_AUTHORIZED = FALSE`. The runner's internal `PILOT_GO_FOR_FULL_H8_CONTINUATION` field is not an authorization and is overridden by the canonical hard gate.

## Six-hour video1 evidence

The formal corrected video1 v2 builder completed all 8,995 records after `06:55:17.54` (`2026-10-07T04:15:18Z`–`11:10:35Z`, 21.67 records/min). Its integrity/provenance passed, but its runtime semantic gate was `NO_GO` (runtime feature parity max error `0.0060040`, 225 candidate mismatches, stability failure). The evidence is therefore proof of the measured six-hour run and durable artifact, not a valid final controller result.

Canonical evidence files are published on the audit branch under:

- `reports/JEV_RNG_V4/VIDEO01_SIX_HOUR_RUNTIME_EVIDENCE_20261007.json`
- `docs/JEV_V2_LIVE_PROGRESS_AND_SIX_HOUR_EVIDENCE_20261007.md`
- `reports/JEV_RNG_V4/VIDEO01_POST_DIAGNOSTIC_AUDIT_20261007.json`

Raw JSONL/checkpoints remain local and are intentionally not committed to GitHub.
