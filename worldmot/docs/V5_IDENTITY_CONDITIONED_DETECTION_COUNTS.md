Native detection counts can depend on predicted identity

This is a generated two-frame, one-camera evaluator mechanism check. It is not a VisionTrack experiment or a V5 tracking score. No real image, GT or prediction is read, no neural model runs, and no formal metric or threshold changes.

The actual B0/B1 formal comparison preserves all548,583 detection candidates. Nevertheless, B1 has7 fewer TrackEval CLEAR TP,7 more FP/FN, and0.1960 percentage points higher HOTA DetA. Those differences alone do not establish a detector change or a detection improvement.

The pinned [CLEAR implementation](https://github.com/LYQ1107/WWW/blob/2f7ca2ca01215b6061dfc9400de3e6cdab4abdca/TrackEval/trackeval/metrics/clear.py#L79) first gives a1000-point matching preference to the prior timestep's identity, then adds IoU and applies the0.5 eligibility threshold. The Hungarian assignment can consequently select fewer eligible pairs to preserve a prior identity. The pinned [HOTA implementation](https://github.com/LYQ1107/WWW/blob/2f7ca2ca01215b6061dfc9400de3e6cdab4abdca/TrackEval/trackeval/metrics/hota.py#L84) multiplies current IoU by global track-identity alignment before assignment; its DetA is therefore also conditioned on identity matching.

The generated fixture uses two GT boxes at both timesteps: `[0,0,10,10]` and `[4,0,14,10]`, in xyxy coordinates. Prediction boxes at the first time equal these boxes. At the second time they are `[2,0,12,10]` and `[-2,0,8,10]`. Both cases use exactly those boxes and the resulting identical IoU matrices:

```text
time1: [[1,   3/7], [3/7, 1  ]]
time2: [[2/3, 2/3], [2/3, 1/4]]
```

Only the second timestamp's tracker IDs change. The unmodified native implementations produce:

| Generated case | IDs at time1 | IDs at time2 | CLEAR TP | FP | FN | IDSW | HOTA DetA (%) |
|---|---|---|---:|---:|---:|---:|---:|
| Fixed box-order IDs | `[0,1]` | `[0,1]` | 3 | 1 | 1 | 0 | 62.1053 |
| Swapped IDs at time2 | `[0,1]` | `[1,0]` | 4 | 0 | 0 | 0 | 78.9474 |

These are synthetic evaluator outputs, not B0/B1 model performance. They establish a possible mechanism with genuine box IoUs. They do not identify the actual7 TEST differences; that remains part of the registered full prediction analysis, with no changes to predictions or model parameters.

The executable check is `worldmot/analysis/check_worldobject_v5_identity_conditioned_metrics.py`; its output is `worldmot/results/v5_identity_conditioned_metrics_check.json`. It verifies the same source hashes recorded in formal scoring:

- CLEAR: `349daafdbb009f6cd71cf66ff1c9b8074940cd3310fcd663151d644b7ff0e5e3`.
- HOTA: `421cb37dd492ecf2591114ea7e4a0d006cd0401225551717fd0b50473c8505d5`.

Frozen detector preservation is established by input/output checks, independently of these matching-conditioned metrics. B1's aggregate identity regression remains unchanged and is not explained away by this mechanism check.
