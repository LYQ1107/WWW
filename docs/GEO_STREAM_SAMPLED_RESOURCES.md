# Passive sampled resource evidence

Each completed finite-root probe is verified against sealed raw CSV hashes. Device utilization is sampled for the whole physical GPU; per-PID RSS is not unique physical RAM. Sample maxima are not lifetime peaks, and sample means are unweighted. Multiple probes overlap and are not independent or additive. Missing final probe artifacts remain pending.

| Probe | GPU | Samples | Mean sampled utilization (%) | Sampled device memory maximum (MiB) | Samples with compute outside named roots |
|---|---:|---:|---:|---:|---:|
| original | 0 | 5709 | 23.278 | 6535.000 | 797 |
| original | 4 | 5709 | 23.704 | 6525.000 | 1358 |
| original | 5 | 5709 | 21.631 | 6511.000 | 1 |
| original | 6 | 5709 | 23.949 | 8510.000 | 19 |
| balanced | 0 | 1897 | 18.429 | 6535.000 | 0 |
| balanced | 4 | 1897 | 25.326 | 6511.000 | 1 |
| balanced | 5 | 1897 | 7.823 | 6451.000 | 0 |
| balanced | 6 | 1897 | 17.384 | 6511.000 | 0 |

original: 2026-10-15T05:35:18.695353+00:00 to 2026-10-15T09:08:05.229524+00:00; 5709 sample groups, observed span 12766.534s; requested interval 2.0s, maximum observed gap 3.007s. Process records: 74. Exact identity/tag/role and per-PID values remain in remaining_job_sampled_resources_v1.json.


balanced: 2026-10-15T08:17:30.128007+00:00 to 2026-10-15T09:27:37.785950+00:00; 1897 sample groups, observed span 4207.658s; requested interval 2.0s, maximum observed gap 2.830s. Process records: 26. Exact identity/tag/role and per-PID values remain in balanced_handoff_sampled_resources_v1.json.

Pending finalized probes: none.

Compute outside a named root set does not identify an unrelated user: replacement roots and discovery timing can also produce these counts. Original/new root sets and time windows must be read with the handoff record. Already-running jobs have partial coverage; later appearance does not establish full lifetime coverage. Per-process allocator peaks and cache/index payloads have separate scopes in resource_ledger_v1.json.
