# VGGT+GMT report (automatically generated; research in progress)

SCIENTIFIC_SUCCESS: not established. ENGINEERING_COMPLETE: false. PAPER_READY: false.

Tables below are read from actual evaluation files. Raw-GT diagnostic rows cannot substitute for strict evaluation. All input frames are retained. Original GMT full-video postprocessing is offline; no real-time identity latency claimed.

| Method | Video | Strict GT | HOTA | AssA | IDF1 | CVIDF1 | IDSW |
|---|---:|---|---:|---:|---:|---:|---:|
| B0_clean_minimal_full_v2 | 10 | yes | 80.9659 | 81.0414 | 98.9676 | 98.9676 | 0 |
| B0_clean_minimal_full_v2 | 18 | yes | 79.3120 | 81.1202 | 96.6809 | 96.6809 | 5 |
| B0_minimal_full | 1 | yes | 87.8104 | 88.2262 | 97.7207 | 97.7207 | 1 |
| B0_minimal_full | 14 | no: raw diagnostic | 79.9871 | 83.1446 | 95.2027 | 95.1967 | 137 |
| B0_minimal_full | 22 | no: raw diagnostic | 84.5038 | 84.7280 | 97.9279 | 97.9279 | 83 |
| B0_strict_remaining_v1 | 11 | yes | 83.3649 | 79.3051 | 92.1306 | 92.1306 | 6 |
| B0_strict_remaining_v1 | 19 | yes | 77.8135 | 81.6011 | 94.6822 | 94.6822 | 29 |
| B2_l17_sym_w025_full_gpu5 | 1 | yes | 87.8104 | 88.2262 | 97.7207 | 97.7207 | 1 |
| B2_l17_sym_w025_full_gpu5 | 10 | yes | 80.9659 | 81.0414 | 98.9676 | 98.9676 | 0 |
| B2_l17_sym_w025_full_gpu5 | 18 | yes | 79.3120 | 81.1202 | 96.6809 | 96.6809 | 5 |
| B6_sift_w05_k10 | 14 | no: raw diagnostic | 79.9871 | 83.1446 | 95.2027 | 95.1967 | 137 |
| B6_sift_w05_k10 | 22 | no: raw diagnostic | 84.5038 | 84.7280 | 97.9279 | 97.9279 | 83 |
| online_B0_full_v18 | 18 | yes | 78.7789 | 81.0427 | 95.9441 | 95.9441 | 29 |

Required remaining work: frozen feature screening and independent DINO comparison; calibrated residual paired full-video tests; dynamic VGGT geometry validity; actual causal GMT protocol with separate baseline; matched-call-budget scheduler experiments; repeated stability checks and WWW retrieval evidence.

See results/experiment_manifest.json for exact commands, seeds, checkpoint SHA, environment, cache status and external artifact paths. See docs/GEO_STREAM_RESEARCH_LOG.md for commits and failed runs.
