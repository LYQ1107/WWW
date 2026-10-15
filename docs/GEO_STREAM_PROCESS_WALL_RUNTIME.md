# Observed complete process replay cost

actual successful full replay stages: parent monotonic time immediately around launch/wait, including Python startup, model setup, replay and export; excludes inter-stage queue waits and separate GT evaluation.

documented held children and conservative priority-owner lifetime overlaps excluded, even if their loop timing is valid; this can also exclude extra predecessor-wait stages; stage snapshots are immutable completed records within mutable queue manifests; outside-loop difference includes imports, setup, export/provenance and normal host scheduling, not pure model setup; cumulative suite FPS is sum(images)/sum(times), not mean FPS, one descriptive run per input, no stable speed or camera/network capture-latency claim.

| Complete causal strict5 suite | Images | Reported loop images/s | Observed process images/s | Outside-loop seconds |
|---|---:|---:|---:|---:|
| C_dino_head_dense_causal_strict_suite_v1 | 14324 | 4.532699 | 4.424863 | 77.014 |
| C_gmt_head_dense_causal_strict_suite_v1 | 14324 | 6.828683 | 6.586951 | 76.980 |
| causal_B0_seed20261015_strict_suite_v1 | 14324 | 7.093395 | 6.829807 | 77.934 |
| causal_B0_seed20261016_strict_suite_v1 | 14324 | 6.870405 | 6.605047 | 83.760 |
| causal_B5_dense_seed20261014_strict_suite_v1 | 14324 | 2.360659 | 2.329130 | 82.138 |
| causal_B5_dense_seed20261015_strict_suite_v1 | 14324 | 2.152700 | 2.120644 | 100.584 |
| causal_B7_fixed_k10_strict_suite_v1 | 14324 | 4.405587 | 4.293186 | 85.123 |
| causal_B7_fixed_k1_strict_suite_v1 | 14324 | 1.126221 | 1.119196 | 79.831 |
| causal_B7_fixed_k20_strict_suite_v1 | 14324 | 5.059706 | 4.915303 | 83.170 |
| causal_B7_fixed_k5_strict_suite_v1 | 14324 | 3.458327 | 3.396276 | 75.673 |
| causal_motion_total80_strict_suite_v1 | 14324 | 5.638600 | 5.464265 | 81.048 |
| native_prior_seed20261014_strict_suite_v1 | 14324 | 6.992716 | 6.753335 | 72.609 |
| native_prior_seed20261015_strict_suite_v1 | 14324 | 6.837483 | 6.577444 | 82.823 |
| native_prior_seed20261016_strict_suite_v1 | 14324 | 6.565688 | 6.320656 | 84.575 |

Recorded full process rows: 112; excluded full held-dispatcher stages: 6. Missing complete suites are not filled with zero setup cost. Sealed stage/log/manifest hashes and separate full retrieval rows remain in the JSON/CSV.
