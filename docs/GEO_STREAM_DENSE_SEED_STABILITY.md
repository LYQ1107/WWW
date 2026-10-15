# Dense causal seed-factor audit

complete strict5 dense B5/B7 TRAIN-head variation at fixed inference RNG20261014, and dense B7 fixed-primary-head20261014 inference RNG variation; paired causal baseline at the same inference seed.

primary B7 trial is shared between factors, not an independent replicate; incomplete factors do not prove stability; three trials are descriptive, without confidence intervals, seed selection, stable speed claims, unique VGGT/geometry attribution or internal Hungarian decision counts.

| Factor | Method | TRAIN head seed | Actual inference seed | Paired CVIDF1 delta (pp) | IDSW delta | Cross-camera output-pair fixes/harms |
|---|---|---:|---:|---:|---:|---:|
| training_head_seed | B5 | 20261014 | 20261014 | +0.089485 | -101 | 73939/5645 |
| training_head_seed | B5 | 20261015 | 20261014 | +0.084604 | -94 | 73800/3765 |
| training_head_seed | B5 | 20261016 | 20261014 | -0.068334 | -90 | 71712/11926 |
| training_head_seed | B7 | 20261014 | 20261014 | +0.089485 | -101 | 73939/5645 |
| training_head_seed | B7 | 20261015 | 20261014 | +0.084604 | -94 | 73800/3765 |
| inference_rng_seed | B7 | 20261014 | 20261014 | +0.089485 | -101 | 73939/5645 |

Missing full trials remain pending; partial scene results are not substituted for native complete-suite evaluation:

- training_head_seed / B5: 3/3 complete; pending .
- training_head_seed / B7: 2/3 complete; pending causal_B7_dense_seed20261016_strict_suite_v1.
- inference_rng_seed / B7: 1/3 complete; pending causal_B7_primary_head_inference_seed20261015_strict_suite_v1, causal_B7_primary_head_inference_seed20261016_strict_suite_v1.

Separate fixed scene1 complete-video observations (full strict suite: false):

| Method | TRAIN head seed | Inference seed | CVIDF1 | Paired delta (pp) | IDSW | VGGT calls | Numerical geometry gates |
|---|---:|---:|---:|---:|---:|---:|---:|
| B5 | 20261014 | 20261014 | 95.471867 | +0.806994 | 214 | 1864 | 0 |
| B5 | 20261015 | 20261014 | 95.270119 | +0.605245 | 232 | 1864 | 0 |
| B5 | 20261016 | 20261014 | 93.387133 | -1.277740 | 225 | 1864 | 0 |
| B7 | 20261014 | 20261014 | 95.471867 | +0.806994 | 214 | 1864 | 3 |
| B7 | 20261015 | 20261014 | 95.270119 | +0.605245 | 232 | 1864 | 3 |
| B7 | 20261016 | 20261014 | 93.387133 | -1.277740 | 225 | 1864 | 3 |

Scene1 causal baseline CVIDF1=94.664873, IDSW=278. Head-seed variation here does not establish a complete-DEV aggregate or an inference-RNG effect. No selection, retraining or coefficient change follows these observations.

Scene1 feature/joint prediction-file byte equality by TRAIN head: 20261014=True, 20261015=True, 20261016=True. This bounded geometry attribution does not extend missing head-seed full suites or validate GT camera poses.
