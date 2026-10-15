# Real TrackVGGT hard-event cohort

Fresh frozen GMT history, actual detector/solver provenance and native logits; no old teacher vectors, GT-built histories or future observations. Fit scenes5/8; held-out TRAIN12. Each run has400 synchronized groups and a second real40-group OFF equality gate before any GT CSV labels.

The first raw collector called all not-known-correct Top1 predictions errors, including impure reference histories. That label is too broad. Raw files are retained; the table below is authoritative and splits known wrong, ambiguous and unknown cases. Actual Hungarian misses are separate from ranking errors. Fit sampling prioritized reachable not-known-correct queries then native margins; held-out sampling used native margins only and is tested not to require GT fields.

| Scene/scope | Selected | Top1 known correct | Top1 known wrong | Top1 ambiguous | Unknown query | Selected actual primary known error/miss | Patch bins0 /1-3 /4-15 /16+ |
|---|---:|---:|---:|---:|---:|---:|---|
| 5/fit | 500 | 499 | 0 | 0 | 1 | 0/5 | 0 / 9 / 355 / 136 |
| 8/fit | 500 | 63 | 6 | 424 | 7 | 4/17 | 0 / 2 / 460 / 38 |
| 12/validation | 300 | 66 | 18 | 134 | 82 | 0/0 | 171 / 127 / 2 / 0 |

Each Top8 coverage denominator and all-collection solver states remain in the machine audit. Five known-positive queries in held-out TRAIN12 lose all pure positive tracks at Top8 despite a correct observation still existing in an impure track; never relabel those pruned cases as true unmatched. Unmatched/ambiguous proposal labels are masked for supervised positive assignment; they remain reported in evaluation coverage. No teacher efficacy follows from event counts.
