# C2b isolated single-error injection

This report uses event-isolated counterfactual replay over the frozen association cache. Each treatment is paired with a SHAM branch from the same pre-event tracker snapshot. Error rate is wrong/total at the future horizon; delta is treatment minus SHAM, in percentage points.

Treatment: `C2b-INJECTION`; bootstrap seed: `20260930`; resamples: `4000`.

| Horizon | Events | SHAM error | Treatment error | Δ | 95% CI for paired Δ |
|---:|---:|---:|---:|---:|---:|
| +5 | 242 | 3.978% | 4.052% | 0.074% | [-0.011%, 0.139%] |
| +10 | 242 | 4.199% | 4.200% | 0.001% | [-0.048%, 0.052%] |
| +20 | 241 | 3.723% | 3.676% | -0.047% | [-0.121%, 0.027%] |

## Scene-stratified pooled rates

### +5

| Scene | Events | SHAM | Treatment | Δ |
|---|---:|---:|---:|---:|
| 00001garden | 58 | 0.420% | 0.420% | 0.000% |
| 00003garden | 56 | 2.935% | 2.941% | 0.006% |
| 00005garden | 128 | 4.980% | 5.083% | 0.102% |

### +10

| Scene | Events | SHAM | Treatment | Δ |
|---|---:|---:|---:|---:|
| 00001garden | 58 | 0.556% | 0.556% | 0.000% |
| 00003garden | 56 | 2.941% | 2.941% | 0.000% |
| 00005garden | 128 | 5.258% | 5.260% | 0.002% |

### +20

| Scene | Events | SHAM | Treatment | Δ |
|---|---:|---:|---:|---:|
| 00001garden | 58 | 0.551% | 0.549% | -0.002% |
| 00003garden | 56 | 3.191% | 3.191% | 0.000% |
| 00005garden | 127 | 4.568% | 4.505% | -0.063% |

## Executability

Frozen event count: **256**.
Rows with no estimable total by reason: `{"current_target_occupied": 280, "scene_end": 8}`.
The paired estimates above include only events with both SHAM and treatment totals at that horizon.

