# H=8 JEV Three-Way First-Round Validation

This is the speed-only first round: one fixed seed (`20261003`) per method on the same full TRAIN H=8 dataset.
Multi-seed mean/std, ensemble, and robustness metrics are intentionally deferred.
Official TEST data is not read for this protocol.

## Locked protocol

- Seed: `20261003`; sequence-disjoint split is fixed before training.
- Training: `20` epochs, batch `128`, AdamW, LR `0.001`.
- All methods consume identical state vectors, questions, legal actions, target probabilities, and sample weights.
- Temperature scaling is fitted on policy validation only and applied to all three methods.

## First-round validation metrics

| Method | Params | Val NLL | Best-action Accuracy | Brier | ECE | Validation Utility |
|---|---:|---:|---:|---:|---:|---:|
| Learnable Threshold | 33,987 | 0.895657 | 0.780015 | 0.009979 | 0.687612 | 36.589950 |
| Generic MLP | 34,163 | 0.894163 | 0.962432 | 0.008617 | 1.072714 | 36.635554 |
| Full JEV | 34,080 | 0.894633 | 0.999806 | 0.008971 | 1.152154 | 36.644655 |

## Gate for tracking comparison

`jev_beats_both_on_val_utility = True`.
The full tracking comparison is authorized only when this single-seed gate is true.

## Non-learned validation references

| Reference | Best-action Accuracy | Validation Utility |
|---|---:|---:|
| majority_action | 0.999806 | 36.644655 |
| uniform_legal_action | 0.678931 | 36.510651 |

## Reproducibility

- Shared dataset: `/home/liuyeqiang/WWW_jev_rng_v4_runtime/small_h8_training_v6_v7/compact_v1` (SHA-256 `ae3f7dbfca79d45fe55134127edcd03603b2f36f2b9dae208eb0fa603fd050f4`).
- Policy split: `/home/liuyeqiang/WWW_jev_rng_v4_runtime/small_h8_training_v6_v7/policy_split_seed20261003.json` (SHA-256 `f353252fc68ca3f038686b06d20579b7bc9c732b4051506562664b1b8ae0f436`).
- Generated UTC: `2026-10-06T14:54:59.035210+00:00`.
