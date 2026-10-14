# JEV closed-loop fastcheck final decision — 2026-10-14

`JEV_INDEPENDENT_ADVANTAGE_NOT_ESTABLISHED`. The bounded fastcheck is complete; scientific success and original master completion are not established.

All rows below use the same full1034-frame two-camera TRAIN16, one frozen seed. References are exact frozen prior inputs and are separate from historical DEV17–19 three-seed scores. TRAIN16 is gradient-free for new modules but pre-exposed by initializers, so this is development evidence.

| Method | HOTA | AssA | CVIDF1 | IDSW |
|---|---:|---:|---:|---:|
| frozen parent | 69.710 | 69.867 | 87.960 | 1013 |
| frozen threshold | 70.605 | 71.106 | 89.825 | 1002 |
| frozen mlp | 68.536 | 66.989 | 86.763 | 1180 |
| frozen attention | 69.727 | 69.690 | 88.805 | 1103 |
| frozen jev | 68.556 | 67.023 | 86.455 | 1455 |
| attention_D0 | 67.425 | 65.181 | 85.784 | 1279 |
| attention_D1 | 68.892 | 68.073 | 88.038 | 1299 |
| direct_attention_D1 | 63.097 | 57.003 | 74.757 | 1692 |
| direct_jev_D1 | 67.052 | 64.476 | 82.406 | 1093 |
| jev_D0 | 66.525 | 63.674 | 82.574 | 1581 |
| jev_D1 | 66.657 | 64.001 | 83.511 | 1891 |

Data benefit is stronger in ordinary Attention; JEV aggregation did not protect identity stability. Direct JEV lowers switches but increases global-map fragmentation and lowers CVIDF1. Past-certificate duration and complete-video fixed-map duration are separate diagnostics; neither is called a unique physical causal share.

| Live model | Scene FPS | FPS change vs Fixed | Stage2 trial-mean p50/p95 ms | Deployed parameters | New trainable parameters |
|---|---:|---:|---|---:|---:|
| attention_D1 | 1.927 | -29.64% | 156.13/207.81 | 36165044 | 169092 |
| direct_jev_D1 | 2.652 | -3.17% | 90.17/112.17 | 36165107 | 169155 |
| parent | 2.739 | +0.00% | 82.43/103.06 | 35995952 | 0 |
| threshold | 1.962 | -28.35% | 154.88/185.57 | 36126227 | 130275 |

Every reported FPS uses three complete live-image runs on the same physical GPU8, V10032GB, with detector/VFCE/actual Gallery/Bank/assignment and no GT journals/evaluator in the timed loop. Native identity state resets between trials; the loaded model is reused. Common dormant pruning is disclosed and is not a JEV scientific gain. Original instantiated system53515309 parameters; common deployed Fixed35995952; Direct JEV36165107 (net+169155 versus common Fixed); Direct Attention36164979 (net+169027).

GPU budget: 3.091/24 registered GPU-wall hours (includes loading/IO; small manual setup/contract probes are outside this registry, so this is not total active-kernel time). Missing live timings: [].


Full metrics, counts, errors, raw source hashes, CV conventions and identity-time diagnostics are in FINAL_RESULTS.json. N01/N10 are individual changed-row certificates; changed UNKNOWN counterparts are separate, so they are not whole-plan causal correctness claims. Direct methods score all legal IDs, and their raw outside_plans count means not selected, not missing from a top8 proposal set.
