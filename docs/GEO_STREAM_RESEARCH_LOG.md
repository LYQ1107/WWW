# VGGT+GMT research log

## 2026-10-14 initial audit

Independent branch: `research/vggt-geo-stream-mcmt-www27-20261014`.
Working directory: `/data1/liuyeqiang/WWW_vggt_geo_20261014`.
Regenerable runtime: `/data1/liuyeqiang/WWW_vggt_geo_runtime`.

Architecture is initial pure GMT, with missing source files restored from official complete source and only mapper class declaration compatibility fix. Original association/memory/GTA/Hungarian are unchanged. Historical JEV worktrees remain untouched.

Checkpoint SHA256 `cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8` loads strictly with 411 state keys. Config unchanged. Baseline final lock remains pending a fresh complete-video evaluation. Original source evaluator mapping bug is fixed explicitly.

Hardware actually 10 V100-DGXS-32GB; existing workloads on GPUs 1,2,9 are left running. Initial GMT GPU0, VGGT planned GPU3, later independent tasks GPU4/5 only when free, at most four project GPUs. No DDP. V100 mixed precision FP16. Available /data1 space initially ~33GB. Sources <250MB, full VGGT weight download ~4.8GiB, target vectors only; no frame/image copies or dense-token caches.

Scene-family split manifest generated from original TRAIN before screening. Development is not unseen by GMT backbone. Prior official TEST exposure is explicitly true. This project does not read TEST labels to select methods.

Four synthetic correctness tests pass (output order, exact rounded pad mapping/inverse, ROIAlign feature coordinates, special-token removal and subpatch zero coverage). Fresh 60-group two-camera GMT smoke passes. This is truncated offline smoke, not scientific evidence or real-time identity throughput. Manifest records dirty-source origin commit; first clean commit supersedes it and fresh formal runs follow. Full-video paired accuracy and actual OFF equivalence still pending.

Next: complete full minimal-development B0; extract frozen multilevel target features on fixed development pairs; rank identity evidence, compare DINO and GMT errors; integrate only after feature evidence and score calibration.
