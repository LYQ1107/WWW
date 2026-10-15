# Project storage budget snapshot

current project-owned runtime entries including hidden entries, GNU du apparent bytes; symlink targets are not followed; derived repository reports are outside this runtime root.

not an atomic filesystem snapshot, filesystem-allocated bytes or a lifetime storage peak; live artifacts may grow; referenced external dataset/GMT checkpoint/environment storage is not included; source snapshots and frozen weights are separate from compact feature/training artifacts; no dense-token or offline-throughput claim follows from directory sizes.

| Category | Apparent bytes | GiB |
|---|---:|---:|
| compact_features_heads_and_samples | 64395323 | 0.059973 |
| environment_under_runtime | 84348128 | 0.078555 |
| frozen_model_weights | 6244486369 | 5.815631 |
| inference_retrieval_and_evaluation_artifacts | 3756227135 | 3.498259 |
| logs_controls_and_diagnostics | 38387949 | 0.035752 |
| source_snapshots | 2262101937 | 2.106747 |

Total apparent bytes: 12449946841. Exact command, timestamps and all per-entry values are in results/storage_budget_v1.json. No cleanup is performed; the command writes its derived reports and stdout.
