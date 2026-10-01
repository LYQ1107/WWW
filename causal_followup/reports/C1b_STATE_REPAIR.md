# C1b isolated state-repair follow-up

This report uses event-isolated counterfactual replay over the frozen association cache. Each treatment is paired with a SHAM branch from the same pre-event tracker snapshot. Error rate is wrong/total at the future horizon; delta is treatment minus SHAM, in percentage points.

Treatment: `C1b-C`; bootstrap seed: `20260930`; resamples: `4000`.

| Horizon | Events | SHAM error | Treatment error | Δ | 95% CI for paired Δ |
|---:|---:|---:|---:|---:|---:|
| +5 | 60 | 6.890% | 6.800% | -0.091% | [-0.516%, 0.494%] |
| +10 | 60 | 6.369% | 6.551% | 0.182% | [-0.133%, 0.787%] |
| +20 | 60 | 6.058% | 5.967% | -0.090% | [-0.208%, 0.017%] |

## Scene-stratified pooled rates

### +5

| Scene | Events | SHAM | Treatment | Δ |
|---|---:|---:|---:|---:|
| 00001garden | 9 | 2.970% | 2.970% | 0.000% |
| 00003garden | 14 | 7.937% | 7.937% | 0.000% |
| 00005garden | 37 | 7.192% | 7.078% | -0.114% |

### +10

| Scene | Events | SHAM | Treatment | Δ |
|---|---:|---:|---:|---:|
| 00001garden | 9 | 1.887% | 1.887% | 0.000% |
| 00003garden | 14 | 5.738% | 6.557% | 0.820% |
| 00005garden | 37 | 7.003% | 7.118% | 0.115% |

### +20

| Scene | Events | SHAM | Treatment | Δ |
|---|---:|---:|---:|---:|
| 00001garden | 9 | 1.802% | 1.802% | 0.000% |
| 00003garden | 14 | 6.400% | 6.400% | 0.000% |
| 00005garden | 37 | 6.552% | 6.437% | -0.115% |

## Executability

Frozen event count: **71**.
Rows with no estimable total by reason: `{"current_target_occupied": 20, "repair_conflict": 180, "scene_end": 19}`.
The paired estimates above include only events with both SHAM and treatment totals at that horizon.


The companion assignment-only and quarantine branches are included in `FOLLOWUP_STATISTICS.json`.
