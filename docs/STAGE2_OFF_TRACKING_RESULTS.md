# Canonical Stage2 OFF tracking result

Updated: 2026-10-05 UTC

This is a baseline report for the verified Stage2 checkpoint. It is an
evaluation record, not a policy-selection input. No JEV architecture,
threshold, horizon, or checkpoint was selected from these TEST metrics.

## Immutable inputs

- Checkpoint: `/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth`
- Checkpoint SHA256: `cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8`
- Config: `/data1/liuyeqiang/WWW_jev_v2/configs/VISION_test.yaml`
- Config SHA256: `bdaeca71d875e8824c3eaf825967a7ba032514297642a6aabe7ad10a740be94a`
- Split: VisionTrack TEST, 44 per-view sequences / 58,038 images
- Prediction JSON SHA256: `a8d2f5c4cc758a887276e59671831215d36a8949d07486a34d78076aaf3e7b3d`
- TrackEval prepared manifest: `/data1/liuyeqiang/WWW/outputs/research_final_v2/off/canonical_trackeval_test/prepared/manifest.json`
- TrackEval report: `/data1/liuyeqiang/WWW/outputs/research_final_v2/off/canonical_trackeval_test/evaluation/metrics.json`
- Cross-view report: `/data1/liuyeqiang/WWW/outputs/research_final_v2/off/canonical_trackeval_test/crossview_v2/metrics.json`

The raw downloaded GT is retained. TrackEval was run in the explicitly
permissive duplicate-ID audit mode because the downloaded GT contains duplicate
rows in several sequences; no GT rows were changed.

## TrackEval combined metrics

| HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Frag |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8.3140 | 6.4506 | 11.9220 | 8.5297 | -70.2413 | 3,247 | 60,103 |

Counts: `Dets=527,474`, `GT_Dets=580,178`, `IDs=932`, `GT_IDs=685`.

## Cross-view audit

The repository cross-view converter reports `CVIDF1` as TrackEval Identity
IDF1 × 100 and `CVMA` as TrackEval CLEAR MOTA × 100. Its two time-axis
conventions are retained separately:

| Convention | CVIDF1 | CVMA | IDSW | Frag |
| --- | ---: | ---: | ---: | ---: |
| Sequential camera blocks | 8.1854 | -70.2676 | 3,399 | 60,293 |
| Interleaved camera/frame | 8.1854 | -70.9970 | 7,625 | 52,987 |

These are the canonical OFF baseline numbers for the current artifact. They do
not authorize an official JEV claim; the formal selection lock is still absent.
