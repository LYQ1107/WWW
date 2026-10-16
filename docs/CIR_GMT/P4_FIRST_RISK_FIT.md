# First bounded FIT12 risk fit

Both preregistered961-parameter heads trained on CPU with seed17, two full epochs, identical normalization and the same452-step sample-order hash.115,614 certified records were each visited twice per head;53,794 UNKNOWN records were masked, not negative examples. Actual training wall time7.400625s, GPU0. Detector/GTA and all large visual models were not trained. This is FIT12, not final TRAIN24.

| Head | Known wrong edges flagged | Known correct edges flagged | In-sample risk F1 | Accepted UNKNOWN edges flagged |
|---|---:|---:|---:|---:|
| Ordinary MLP | 21 | 64 | 0.162162 | 170 |
| Grouped risk | 24 | 92 | 0.165517 | 373 |

The threshold follows the frozen FIT-only F1 criterion under correct-record FPR<=0.001. These are risk-classification flags on original-state cached inputs, **not certified repairs, harms, actual online VETO counts or evidence of structured-head superiority**. The grouped head also flags more correct and UNKNOWN edges. Both remain unqualified for native online claims.

Actual saved checkpoints reproduce every classification count and share bitwise-identical normalization. Loading preserves Torch RNG. Joining the separately sealed GT-free native policy events distinguishes legal accepted-edge flags (255 MLP/489 grouped) from raw score crossings (1409/1643). Both have1154 high-score queries with no accepted edge: those cannot be vetoed. UNKNOWN accepted edges remain unlabelled, not false positives or successes. Different trigger rates must not be mistaken for an independent architecture effect.

Weights and complete per-person fit diagnostics stay on the server; paths/SHA, compact real results, coverage and CPU ledger are in `TRAINING_RESULTS.json`. Native integration, matched full-video evaluation, later seeds and complete TRAIN24 fit remain pending. No TEST was read.
