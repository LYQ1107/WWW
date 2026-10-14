# Closed-loop training evidence

Fixed last checkpoints, unchanged architecture and frozen identity backbone. All methods see identical state/label pools and identical sampled schedule within a data arm. TRAIN16 is pre-exposed gradient-free diagnostics, not independent validation.

| Architecture | Dataset | New steps | Gradient groups | Source-state net N01−N10 | Rank accuracy |
|---|---|---:|---:|---:|---:|
| attention | D0 | 1000 | 2867 | -16 | 0.7332 |
| attention | D0 | 300 | 2417 | -5 | 0.7508 |
| attention | D1 | 1000 | 3122 | -12 | 0.7514 |
| attention | D1 | 300 | 2589 | -14 | 0.7495 |
| jev | D0 | 1000 | 2867 | -38 | 0.7074 |
| jev | D0 | 300 | 2417 | -54 | 0.6627 |
| jev | D1 | 1000 | 3122 | -28 | 0.7048 |
| jev | D1 | 300 | 2589 | 1 | 0.7413 |

At300 steps JEV D1−D0 is +55 source-state net corrections; Attention D1−D0 is −9. This is a fixed-source-state diagnostic and cannot establish full native benefit or architecture-specific science. All four300 pilots have real ranking-head gradients and pass the preregistered signal eligibility; qualification is not positive-result selection.
