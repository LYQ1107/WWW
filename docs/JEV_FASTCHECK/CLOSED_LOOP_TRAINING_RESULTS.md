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

The one conditional direct-readout repair and ordinary matched control each receive the same D1 schedule for1000 additional updates. Their source-state nets are+11/−21, but native results are HOTA67.052/63.097 and CVIDF1=82.406/74.757. Direct JEV remains below stronger ordinary references. Source-state gains do not establish continuous identity improvement; complete results and both error-duration conventions are reported in FINAL_RESULTS.json.

The five P2 questions have the following bounded answers:

- Current association correctness: source-state JEV net improves−38→−28, but native changed-row N01/N10 goes52/125→112/230. Net certified changed-row corrections worsen−73→−118. These certificates cover individual known rows, not all changed plans.
- Protection of correct identities: JEV N10 increases125→230; retained certified correct actions change48044/50786→48157/50801. This does not support fewer anti-corrections. UNKNOWN changes are181/1507→105/1570 and remain outside the known-row certificates.
- Continuous tracking: JEV HOTA increases+0.132 and AssA+0.327, while IDSW increases310. Complete-video fixed-map inconsistency falls7564→6957 camera-frames, but first-past certified-owner duration increases3582→4372. Both conventions are disclosed; neither proves a unique causal propagation mechanism.
- Ordinary control: Attention HOTA improves+1.467 and AssA+2.892; native N01/N10 changes40/191→54/83. The first-past and complete-video durations fall4169→4077 and5820→4758 camera-frames.
- JEV-specific value: none established. The larger data benefit is in ordinary Attention, and the direct JEV variant remains below strong existing ordinary controls. These are single-seed development comparisons, with no significance or independent-validation claim.

The gradient-group counts above use the actual segment being trained:300 steps for pilots and the additional700 steps for continuation reports. Direct training uses1000 steps and its documented person/camera/block grouping, so its907 groups should not be compared as an IID sample-size ratio to the plan-ranking groups. All sampled histories remain correlated.
