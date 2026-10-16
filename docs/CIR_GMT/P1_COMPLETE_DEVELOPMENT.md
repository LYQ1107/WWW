# Complete C1 development evidence

All 12 preregistered development videos completed real image inference for both Original GMT and C1 (30,702 images per method), from source `3e6d6c5d4f794772070b6cd1e80fc84447e9546a`. No feature cache or teacher was used. Complete metrics and pinned artifacts are in `results/CIR_GMT/DEV_ONLINE_RESULTS.json`.

Strict unmodified evaluation is available on the five predeclared raw-GT-valid videos 1/10/11/18/19. The other seven videos have duplicate raw GT identities; their inference is complete but strict metrics remain null. These five-video scores must never be described as full-12 or full-24 metrics.

| Policy | Raw HOTA | Raw AssA | Raw CVIDF1 | Raw IDSW | Stock HOTA | Stock IDSW | All12 image FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original GMT | 80.748757 | 81.472600 | 95.057962 | 482 | 81.055072 | 41 | 6.236043 |
| C1 | 80.822127 | 81.625552 | 95.251576 | 371 | 81.036274 | 48 | 6.167111 |

C1 has a small raw gain but stock HOTA decreases and stock IDSW increases. This is not comprehensive improvement or final qualification.

The all12 causal-owner diagnostic records 3,377 wrong-to-correct and 5,985 correct-to-wrong observations, across 46 repaired and 56 harmed scene-person dependence groups. These are correlated observations, not thousands of independent successful interventions. On valid-GT video19, wrong-owner observations increase from 1 to 1,785 and the longest observed error streak from 1 to 595 camera-frames. This material harm must not be hidden by pooled scores. The past-owner diagnostic differs from optimal IDF1 identity mapping; UNKNOWN and unanchored cases remain separate, with censoring documented in the JSON.

The frozen FIT12 guard audit separately examined 169,408 native queries: 26 rule SUSPECT outputs, with 4 known wrong accepts, 2 known correct accepts and 20 UNKNOWN. Overall 174 wrong-accept labels in 72 scene-person groups support bounded risk classification, but do not establish that vetoing those edges helps. No guard online benefit, trained-controller benefit, full24 fit or TEST result has been obtained. Continue the registered minimum guard engineering and paired native future checks; do not select C1 as a safe final method.
