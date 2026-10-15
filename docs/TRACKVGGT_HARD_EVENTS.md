# Real TrackVGGT hard-event cohort

Fresh frozen GMT history, actual detector/solver provenance and native logits; no old teacher vectors, GT-built histories or future observations. Fit scenes5/8; held-out TRAIN12. Each run has400 synchronized groups and a second real40-group OFF equality gate before any GT CSV labels.

The first raw collector called all not-known-correct Top1 predictions errors, including impure reference histories. That label is too broad. Raw files are retained; the table below is authoritative and splits known wrong, ambiguous and unknown cases. Actual Hungarian misses are separate from ranking errors. Fit sampling prioritized reachable not-known-correct queries then native margins; held-out sampling used native margins only and is tested not to require GT fields.

| Scene/scope | Selected | Top1 known correct | Top1 known wrong | Top1 ambiguous | Unknown query | Selected actual primary known error/miss | Patch bins0 /1-3 /4-15 /16+ |
|---|---:|---:|---:|---:|---:|---:|---|
| 5/fit | 500 | 499 | 0 | 0 | 1 | 0/5 | 0 / 9 / 355 / 136 |
| 8/fit | 500 | 63 | 6 | 424 | 7 | 4/17 | 0 / 2 / 460 / 38 |
| 12/validation | 300 | 66 | 18 | 134 | 82 | 0/0 | 171 / 127 / 2 / 0 |

Each Top8 coverage denominator and all-collection solver states remain in the machine audit. Five known-positive queries in held-out TRAIN12 lose all pure positive tracks at Top8 despite a correct observation still existing in an impure track; never relabel those pruned cases as true unmatched. Unmatched/ambiguous proposal labels are masked for supervised positive assignment; they remain reported in evaluation coverage. No teacher efficacy follows from event counts.

## Locked expanded fit cohort

The prespecified legal fit expansion to scenes4/6/7 is complete, with separate fresh OFF40 parity in all three collectors. The final pre-teacher fit is1000 unique queries across scenes4/5/6/7/8; held-outTRAIN12 remains the original300 native-margin-selected queries. The fixed rule retains130 reachable known ranking/primary wrong-or-missed events and19 known Top1 ranking errors; these are different counts, not independent corrections. Per-fit-scene sample counts: {'4': 200, '5': 179, '6': 199, '7': 227, '8': 195}. No teacher result was used in selection. Exact indices/raw artifact hashes are in results/trackvggt_expanded_cohort_v2.json.

Real zero-residual adapter deployment additionally passes40 synchronized groups/80 images and79 association calls, with exactly the same prediction-file bytes as native GMT. No teacher/student checkpoint or VGGT is loaded in that gate; it is not learned deployment evidence. Remaining teacher/student/cycle claims require training and actual paired evaluation.
