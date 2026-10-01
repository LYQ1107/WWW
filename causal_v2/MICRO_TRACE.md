# C2c micro target trace

The event set is the first 10 C2b manifest events in lexicographic event-key order.

## Micro gate

**PASS: target-centric tracking confirmed.** For every executable injection,
the event-frame treated detection is `p_wrong` while SHAM is the frozen
baseline identity (`p_correct := p_baseline` because the C2b manifest has no
separate `p_correct` field). The future rows report only the treated GT's
matched observations; unmatched horizons are explicitly `target_not_observable`
and are not converted to zero error. Seven of the ten injections were
executable. Three were explicitly skipped as `baseline_anchor_mismatch`
(`00001garden|179|1|11`, `00001garden|196|1|7`, and
`00001garden|208|1|3`); the replay never forced those frozen events to a
different current identity.

## 00001garden|122|1|7 — C2c-INJECTION

| Time | Treated target track IDs across visible views | target error |
|---:|---|---:|
| t | `9` | — |
| t+1 | `[37]` | `0.0` |
| t+2 | `[37]` | `0.0` |
| t+5 | `[37]` | `0.0` |
| t+10 | `[37]` | `0.0` |
| t+20 | `[37]` | `0.0` |

## 00001garden|122|1|7 — SHAM

| Time | Treated target track IDs across visible views | target error |
|---:|---|---:|
| t | `37` | — |
| t+1 | `[37]` | `0.0` |
| t+2 | `[37]` | `0.0` |
| t+5 | `[37]` | `0.0` |
| t+10 | `[37]` | `0.0` |
| t+20 | `[37]` | `0.0` |

## 00001garden|125|1|0 — C2c-INJECTION

| Time | Treated target track IDs across visible views | target error |
|---:|---|---:|
| t | `9` | — |
| t+1 | `[7, 7]` | `0.0` |
| t+2 | `[7, 7]` | `0.0` |
| t+5 | `[7, 7]` | `0.0` |
| t+10 | `[7, 7]` | `0.0` |
| t+20 | `[7, 7]` | `0.0` |

## 00001garden|125|1|0 — SHAM

| Time | Treated target track IDs across visible views | target error |
|---:|---|---:|
| t | `7` | — |
| t+1 | `[7, 7]` | `0.0` |
| t+2 | `[7, 7]` | `0.0` |
| t+5 | `[7, 7]` | `0.0` |
| t+10 | `[7, 7]` | `0.0` |
| t+20 | `[7, 7]` | `0.0` |

## 00001garden|125|1|6 — C2c-INJECTION

| Time | Treated target track IDs across visible views | target error |
|---:|---|---:|
| t | `9` | — |
| t+1 | `[1]` | `0.0` |
| t+2 | `[1]` | `0.0` |
| t+5 | `[1]` | `0.0` |
| t+10 | `[1]` | `0.0` |
| t+20 | `[1]` | `0.0` |

## 00001garden|125|1|6 — SHAM

| Time | Treated target track IDs across visible views | target error |
|---:|---|---:|
| t | `1` | — |
| t+1 | `[1]` | `0.0` |
| t+2 | `[1]` | `0.0` |
| t+5 | `[1]` | `0.0` |
| t+10 | `[1]` | `0.0` |
| t+20 | `[1]` | `0.0` |

## 00001garden|15|1|9 — C2c-INJECTION

| Time | Treated target track IDs across visible views | target error |
|---:|---|---:|
| t | `10` | — |
| t+1 | `[12]` | `0.0` |
| t+2 | `[]` | `None` |
| t+5 | `[]` | `None` |
| t+10 | `[12]` | `0.0` |
| t+20 | `[]` | `None` |

## 00001garden|15|1|9 — SHAM

| Time | Treated target track IDs across visible views | target error |
|---:|---|---:|
| t | `12` | — |
| t+1 | `[12]` | `0.0` |
| t+2 | `[]` | `None` |
| t+5 | `[]` | `None` |
| t+10 | `[12]` | `0.0` |
| t+20 | `[]` | `None` |

## 00001garden|179|1|11 — C2c-INJECTION

| Time | Treated target track IDs across visible views | target error |
|---:|---|---:|
| t | `10` | — |
| t+1 | `[]` | `None` |
| t+2 | `[]` | `None` |
| t+5 | `[]` | `None` |
| t+10 | `[]` | `None` |
| t+20 | `[]` | `None` |

## 00001garden|179|1|11 — SHAM

| Time | Treated target track IDs across visible views | target error |
|---:|---|---:|
| t | `10` | — |
| t+1 | `[10]` | `1.0` |
| t+2 | `[10]` | `1.0` |
| t+5 | `[10]` | `1.0` |
| t+10 | `[10]` | `1.0` |
| t+20 | `[10]` | `1.0` |

## 00001garden|196|1|7 — C2c-INJECTION

| Time | Treated target track IDs across visible views | target error |
|---:|---|---:|
| t | `13` | — |
| t+1 | `[]` | `None` |
| t+2 | `[]` | `None` |
| t+5 | `[]` | `None` |
| t+10 | `[]` | `None` |
| t+20 | `[]` | `None` |

## 00001garden|196|1|7 — SHAM

| Time | Treated target track IDs across visible views | target error |
|---:|---|---:|
| t | `13` | — |
| t+1 | `[13]` | `1.0` |
| t+2 | `[13]` | `1.0` |
| t+5 | `[13]` | `1.0` |
| t+10 | `[13]` | `1.0` |
| t+20 | `[13]` | `1.0` |

## 00001garden|207|1|2 — C2c-INJECTION

| Time | Treated target track IDs across visible views | target error |
|---:|---|---:|
| t | `1` | — |
| t+1 | `[7, 7]` | `0.0` |
| t+2 | `[7, 7]` | `0.0` |
| t+5 | `[7, 7]` | `0.0` |
| t+10 | `[7, 7]` | `0.0` |
| t+20 | `[87, 7]` | `0.5` |

## 00001garden|207|1|2 — SHAM

| Time | Treated target track IDs across visible views | target error |
|---:|---|---:|
| t | `7` | — |
| t+1 | `[7, 7]` | `0.0` |
| t+2 | `[7, 7]` | `0.0` |
| t+5 | `[7, 7]` | `0.0` |
| t+10 | `[7, 7]` | `0.0` |
| t+20 | `[87, 7]` | `0.5` |

## 00001garden|207|1|3 — C2c-INJECTION

| Time | Treated target track IDs across visible views | target error |
|---:|---|---:|
| t | `1` | — |
| t+1 | `[5]` | `0.0` |
| t+2 | `[5]` | `0.0` |
| t+5 | `[5]` | `0.0` |
| t+10 | `[5]` | `0.0` |
| t+20 | `[5]` | `0.0` |

## 00001garden|207|1|3 — SHAM

| Time | Treated target track IDs across visible views | target error |
|---:|---|---:|
| t | `5` | — |
| t+1 | `[5]` | `0.0` |
| t+2 | `[5]` | `0.0` |
| t+5 | `[5]` | `0.0` |
| t+10 | `[5]` | `0.0` |
| t+20 | `[5]` | `0.0` |

## 00001garden|208|1|3 — C2c-INJECTION

| Time | Treated target track IDs across visible views | target error |
|---:|---|---:|
| t | `52` | — |
| t+1 | `[]` | `None` |
| t+2 | `[]` | `None` |
| t+5 | `[]` | `None` |
| t+10 | `[]` | `None` |
| t+20 | `[]` | `None` |

## 00001garden|208|1|3 — SHAM

| Time | Treated target track IDs across visible views | target error |
|---:|---|---:|
| t | `52` | — |
| t+1 | `[52]` | `1.0` |
| t+2 | `[52]` | `1.0` |
| t+5 | `[52]` | `1.0` |
| t+10 | `[52]` | `1.0` |
| t+20 | `[52]` | `1.0` |

## 00001garden|248|1|4 — C2c-INJECTION

| Time | Treated target track IDs across visible views | target error |
|---:|---|---:|
| t | `7` | — |
| t+1 | `[5]` | `0.0` |
| t+2 | `[5]` | `0.0` |
| t+5 | `[5]` | `0.0` |
| t+10 | `[5]` | `0.0` |
| t+20 | `[5]` | `0.0` |

## 00001garden|248|1|4 — SHAM

| Time | Treated target track IDs across visible views | target error |
|---:|---|---:|
| t | `5` | — |
| t+1 | `[5]` | `0.0` |
| t+2 | `[5]` | `0.0` |
| t+5 | `[5]` | `0.0` |
| t+10 | `[5]` | `0.0` |
| t+20 | `[5]` | `0.0` |
