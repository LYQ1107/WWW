# Closed-loop training evidence

Fixed last checkpoints, unchanged architecture and frozen identity backbone. All methods see identical state/label pools and identical sampled schedule within a data arm. TRAIN16 is pre-exposed gradient-free diagnostics, not independent validation.

| Architecture | Dataset | New steps | Gradient groups | Source-state net N01−N10 | Rank accuracy |
|---|---|---:|---:|---:|---:|
| attention | D0 | 1000 | 2867 | -16 | 0.7332 |
| attention | D0 | 300 | 2417 | -5 | 0.7508 |
| attention | D1 | 1000 | 3122 | -12 | 0.7514 |
| attention | D1 | 300 | 2589 | -14 | 0.7495 |
| direct_attention | D1 | 1000 | 907 | -21 | N/A: direct readout |
| direct_jev | D1 | 1000 | 907 | 11 | N/A: direct readout |
| jev | D0 | 1000 | 2867 | -38 | 0.7074 |
| jev | D0 | 300 | 2417 | -54 | 0.6627 |
| jev | D1 | 1000 | 3122 | -28 | 0.7048 |
| jev | D1 | 300 | 2589 | 1 | 0.7413 |

At300 steps JEV D1−D0 is +55 source-state net corrections; Attention D1−D0 is −9. This is a fixed-source-state diagnostic and cannot establish full native benefit or architecture-specific science. All four300 pilots have real ranking-head gradients and pass the preregistered signal eligibility; qualification is not positive-result selection.

At the fixed1000-step checkpoints the source-state JEV nets are−38/−28 forD0/D1; Attention nets are−16/−12. Full cold-start1034-frame TRAIN16 gives JEV HOTA66.525/66.657 and IDSW1581/1891, while Attention gives HOTA67.425/68.892 and IDSW1279/1299. Data aggregation helps ordinary Attention substantially more on HOTA; it does not protect JEV identity stability. Both D1 models remain below their frozen original respective initializers. No cross-seed or independent-video benefit is claimed.

The one conditional direct-readout repair and ordinary matched control each receive the same D1 schedule for1000 additional updates. Their source-state nets are+11/−21, but native results are HOTA67.052/63.097 and CVIDF182.406/74.757. Direct JEV remains below stronger ordinary references. Source-state gains do not establish continuous identity improvement; complete results and both error-duration conventions are reported in FINAL_RESULTS.json.
