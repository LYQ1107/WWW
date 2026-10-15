# Fresh native system results

Status: ACTUAL_NATIVE_SYSTEM_PROGRESS_SNAPSHOT. Pending full cases: 1; pending fresh strict5 metrics: 1.

Actual fresh full image replay only. OFF/Prior prefixes and completed full outputs are checked against their own original standalone references. No TrackVGGT teacher/student is loaded.

| Method | Configured streams | Images | Whole process seconds | Whole process image FPS | Round P95 ms | Query response P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| gmt_off | 2 | 2058 | 312.37 | 6.588 | 332.48 | 2.436 |
| native_fixed_prior | 2 | 2058 | 322.37 | 6.384 | 338.02 | 2.360 |
| gmt_off | 4 | 3922 | 539.50 | 7.270 | 593.87 | 4.701 |
| native_fixed_prior | 4 | 3922 | 534.62 | 7.336 | 573.23 | 4.179 |
| gmt_off | 10 | 14324 | 1926.61 | 7.435 | 1402.00 | 10.597 |
| native_fixed_prior | 10 | PENDING | PENDING | PENDING | PENDING | PENDING |

| Method | Streams | Requests | GT matched | Gallery eligible | Conditional R1 | Conditional R5 | Full MRR | R1 success /all requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gmt_off | 2 | 598 | 577 | 309 | 0.9773 | 1.0000 | 0.9879 | 0.5050 |
| native_fixed_prior | 2 | 598 | 577 | 309 | 0.9773 | 1.0000 | 0.9887 | 0.5050 |
| gmt_off | 4 | 826 | 798 | 494 | 0.9777 | 0.9960 | 0.9863 | 0.5847 |
| native_fixed_prior | 4 | 826 | 798 | 494 | 0.9777 | 0.9960 | 0.9868 | 0.5847 |
| gmt_off | 10 | 3080 | 3016 | 2465 | 0.9229 | 0.9785 | 0.9463 | 0.7386 |
| native_fixed_prior | 10 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

Conditional Recall/MRR denominator: GT-matched detector queries with a positive other-camera clip in the original full native gallery. All-request success retains unmatched and ineligible detector queries. An identical count does not prove identical eligible members; the separate system_query_pairs audit verifies physical query/GT cohorts, gains/losses and common-query fixes/harms.

| Method | Streams | CUDA allocator peak incl. setup, bytes | Host peak RSS, bytes | Index peak payload, bytes |
|---|---:|---:|---:|---:|
| gmt_off | 2 | 772823040 | 3889545216 | 1168118 |
| native_fixed_prior | 2 | 777539072 | 3874754560 | 1186878 |
| gmt_off | 4 | 818240000 | 3951095808 | 1615405 |
| native_fixed_prior | 4 | 829643264 | 3915886592 | 1582858 |
| gmt_off | 10 | 1259291648 | 4025229312 | 3581241 |
| native_fixed_prior | 10 | PENDING | PENDING | PENDING |

Each run uses one actual V100-DGXS32GB. Configured2/4/10 streams are1/2/5 separate two-camera scenes, interleaved by replay round. Active membership decreases as unequal-length scenes finish; no constant10-active-stream or Internet-deployment claim. Scene/target mix also changes across stream counts, so this is not an isolated scaling curve with a fixed workload. Concurrent producers share a host, CPU and file storage, so one observed timing per case is not a repeated isolated speed comparison.

Whole process wall includes Python/import/setup/replay/export; post-run GT evaluators are separate. Identical synchronization profiling is charged to both methods. Read/preprocess, tracker and query compute are timed separately; remaining loop time combines publication, serialization and control and is not an isolated publication benchmark. Query response latency excludes network.

Full strict5 metrics are in full_tracking_metrics.csv, with inherited native/Prior inference seeds explicitly separate from fresh runs. Frag for inherited rows is extracted from retained raw CLEAR after checking its HOTA/IDF1/IDSW association; it is not a new tracking execution. Fresh Frag comes from the new raw combined evaluator output.

Raw run, parity, metric, process log and queue hashes are bound in the JSON/CSV results. The native-only systems evidence does not establish a TrackVGGT method benefit, student deployment, scientific success or WWW paper readiness.
