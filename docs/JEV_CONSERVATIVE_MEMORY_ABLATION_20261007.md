# Conservative memory-action ablation — 2026-10-07

This is a screening diagnostic, not a final method selection or paper result.
It uses the completed current-head segmented H=8 data and the existing
calibrated checkpoints. It does not rebuild data, retrain GMT, rerun the
official test, or authorize Full H8.

## Experimental change

Only `MEMORY_DECISION` is overridden to the GMT OFF action `WRITE_MEMORY`.
`MATCH_DECISION` and `REACTIVATION_DECISION` remain controlled by each loaded
checkpoint. The experiment is bound to:

```text
runner commit:       1db16ec
semantic base:       1d2711e80ac5fa00806fd9eed30e90cd51df6b30
video01 records:     8995
records SHA-256:     a174399cb979d5d1f9dc41366a4f07f6299c159c68e10e19222fedc656db7611
controller dataset:  8496 records
dataset manifest:    3a0c5b05956d62ea09837678261559ac94d65ba77ba8eb16710c183f724f5df5
policy split:        bbdac3cff7c46b20c707dd278c526704a99ede973cc1ffde31b11172df9efd91
seed:                20261003
```

## Tracking result

| Method | HOTA | AssA | IDF1 | MOTA | IDSW | ΔAssA vs OFF |
|---|---:|---:|---:|---:|---:|---:|
| GMT OFF | 86.523 | 85.149 | 94.642 | 89.579 | 282 | 0.000 |
| Threshold + memory safety | 80.034 | 72.917 | 83.995 | 78.453 | 771 | -12.232 |
| MLP + memory safety | 57.900 | 38.481 | 59.180 | 62.708 | 1463 | -46.668 |
| JEV + memory safety | 85.873 | 83.978 | 92.827 | 95.586 | 18 | -1.172 |

Compared with the unmodified current-head JEV, the memory safety gate improves
JEV by:

```text
ΔHOTA  +0.801
ΔAssA  +1.518
ΔIDF1  +0.852
ΔMOTA   0.000
ΔIDSW   0
```

JEV still changes 26 MATCH decisions to `REASSOCIATE` and 6 to `START_NEW`,
so the remaining association loss is not explained by memory writes alone.

## Gate status

- GMT OFF runtime feature parity: PASS, 8995/8995, no missing records, no
  off-action mismatch.
- Reactivation coverage: PASS, 174/174 decisions and candidate evaluations.
- This is still `PARTIAL_DIAGNOSIS_NO_GO`; JEV AssA and HOTA remain below GMT
  OFF. `FULL_H8_AUTHORIZED=false`.

## Interpretation and next step

The result confirms that the 100% MEMORY tie rate in the current policy data
is a real contributor to learned-policy state shift: forcing the baseline
memory action recovers part of the tracking loss. It does not validate the
current controller as a final method.

The next controlled experiment should select a tie-aware target or explicit
conservative gate for MATCH/REASSOCIATE on train/validation data, then rerun a
fresh held-out small gate. The original current-head result, this ablation,
and the earlier legacy positive bundle must remain separate bindings.

Machine-readable details are in
[`CONSERVATIVE_MEMORY_ABLATION_20261007.json`](../reports/JEV_RNG_V4/CONSERVATIVE_MEMORY_ABLATION_20261007.json).
