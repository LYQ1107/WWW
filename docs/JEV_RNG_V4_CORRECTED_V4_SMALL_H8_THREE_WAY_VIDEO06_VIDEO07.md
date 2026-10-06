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
| Learnable Threshold | 33,987 | 0.895639 | 0.916925 | 0.009966 | 0.479910 | 36.624177 |
| Generic MLP | 34,163 | 0.896707 | 0.704880 | 0.010381 | 0.284270 | 36.467661 |
| Full JEV | 34,080 | 0.894535 | 0.999806 | 0.008881 | 0.572758 | 36.644655 |

## Gate for tracking comparison

`jev_beats_both_on_val_utility = True`.
`jev_beats_majority_reference = False`; `jev_matches_majority_reference = True`.
A majority tie is not evidence of a meaningful learned-policy advantage; runtime parity and reactivation coverage are separate hard gates.

## Non-learned validation references

| Reference | Best-action Accuracy | Validation Utility |
|---|---:|---:|
| majority_action | 0.999806 | 36.644655 |
| uniform_legal_action | 0.678931 | 36.510651 |

## Reproducibility

- Shared dataset: `/home/liuyeqiang/WWW_jev_rng_v4_runtime/small_h8_training_v4_canonical_features/compact_v1` (SHA-256 `cabedc4fa2df26a605c9b0523b35ea75b7c1763d907aa16442101e3bec07565c`).
- Policy split: `/home/liuyeqiang/WWW_jev_rng_v4_runtime/small_h8_training_v4_canonical_features/policy_split_seed20261003.json` (SHA-256 `869a975d024441ce1135f5b90f095c9ea5e98d5b34d10be535a31024c4bdafe7`).
- Generated UTC: `2026-10-06T20:46:55.760630+00:00`.
