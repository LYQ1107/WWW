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

## Phase 0/1 follow-up

Initial clean commit `8a39a3ec4274866c85fc8f2798e838e6f93c7d22` pushed. Fixed 30 TRAIN development synchronized samples extracted fresh; first extraction failed because postprocess mutated an inference tensor outside InferenceMode. Corrected by no-grad context, retry `screen_v1r2` completes. No failed artifact reused.

Geometry reference: 120 fixed settings (30 pairs x ORB/SIFT x target masking on/off). Masked SIFT valid 17/30, masked ORB 7/30. Validity uses background only, not GT. SIFT can fail or retain confusing candidate scores; no tracking improvement claim.

Full garden video01 baseline: HOTA 87.8104, AssA 88.2262, IDF1/CVIDF1 97.7207, IDSW=1, strict raw-GT TrackEval passes. Full football video14 has 578+4 conflicting GT identity rows; strict TrackEval refuses. All raw rows preserved. Permissive diagnostic evaluation is separately labeled; it cannot be represented as valid strict evaluation. An all-TRAIN GT quality audit precedes method selection.

Residual hook is optional in pre-softmax proposal logits. GTA/Hungarian/detector/default memory untouched. Feature scores require scalar calibration on association TRAIN; unsupported memory receives neutral evidence. Geometry reference uses same-time coordinates only, never applies an old F to unpropagated current points. Defaults and zero weights bypass extraction completely. Fresh OFF and enabled-zero 60-group predictions exactly equal pure-GMT smoke JSON including IDs and ordering. Complete-video parity remains pending. Geometry matrix direction, camera-from-world synthetic projection, degeneracy, sparse scheduler prefix/budget, bounded worker and residual tests pass. These unit checks are not evidence of strict-online GMT execution; original postprocessing remains offline.

## GT quality protocol and clean screening

Second pushed commit: `624ba2cfcc` (use Git for full SHA). All-TRAIN audit finds conflicting within-frame GT identities in videos 2,3,14,15,16,20,22,23,24. No labels repaired. Split v2 freezes strict development subset by this structural property BEFORE any VGGT model screening results: videos 1,10,11,18,19. All original development videos remain listed, and contaminated videos are evaluated as explicitly separate raw-GT diagnostics. Minimal strict screen uses [1,10,18] and fixed linspace timestamps. Initial [1,14,22] samples are retained as diagnostics and are not used for scalar calibration. Association training samples [5,8,12] belong to separate original-TRAIN scene families and have unique raw GT.

B0 original full-video runs for [1,14,22] complete, all image frames retained. Fresh clean B0 [10,18] is running. Predeclared reference experiment B6_sift_w05_k10 [14,22], weight 0.5 and interval 10, begins before seeing any enhanced tracking metrics. This is a traditional masked-background reference, not a VGGT DCGA claim. Original memory score path is explicitly neutral. Raw-diagnostic evaluation retry used a separate output directory after strict audit left an empty directory; no completed result overwritten.

VGGT official research weights and independent public DINOv2 ViT-L/14-register weights are downloaded into runtime only. All geometry code and matrices are original small implementations; no StreamVGGT/FastVGGT code copied.
