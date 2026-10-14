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

Second pushed commit: `624ba2cf89bb0443866863b06e19d8422d76c88b`. All-TRAIN audit finds conflicting within-frame GT identities in videos 2,3,14,15,16,20,22,23,24. No labels repaired. Split v2 freezes strict development subset by this structural property BEFORE any VGGT model screening results: videos 1,10,11,18,19. All original development videos remain listed, and contaminated videos are evaluated as explicitly separate raw-GT diagnostics. Minimal strict screen uses [1,10,18] and fixed linspace timestamps. Initial [1,14,22] samples are retained as diagnostics and are not used for scalar calibration. Association training samples [5,8,12] belong to separate original-TRAIN scene families and have unique raw GT.

B0 original full-video runs for [1,14,22] complete, all image frames retained. Fresh clean B0 [10,18] is running. Predeclared reference experiment B6_sift_w05_k10 [14,22], weight 0.5 and interval 10, begins before seeing any enhanced tracking metrics. This is a traditional masked-background reference, not a VGGT DCGA claim. Original memory score path is explicitly neutral. Raw-diagnostic evaluation retry used a separate output directory after strict audit left an empty directory; no completed result overwritten.

VGGT official research weights and independent public DINOv2 ViT-L/14-register weights are downloaded into runtime only. All geometry code and matrices are original small implementations; no StreamVGGT/FastVGGT code copied.

## Frozen VGGT smoke, camera order failure and recovery

VGGT original checkpoint download completes (5,026,874,952 bytes) and strict architecture load succeeds. Two clean garden sample pairs run jointly, reversed and singly. Observed aggregator length24, cached levels4/11/17/23 shape [1,2,1374,2048], patch_start_idx5, patch grid37x37. Warm joint inference about0.98s on this V100, first invocation4.28s; not total pipeline throughput. Target only storage.

Camera order is NOT invariant: observed maximum candidate cosine change0.2075 after reversing views and restoring camera IDs. Consequently raw joint scores are diagnostic only until this is addressed. Added opt-in symmetric representation: concatenate per-target vectors from both joint input orders with 1/sqrt(2) scaling, yielding the mean of the two cosine matrices. This doubles actual VGGT calls and is charged in counters and latency. Scalar calibration must match the same symmetric representation. Full clean screening already underway retains raw, single and symmetric score comparisons; subsequent TRAIN extraction includes both order target vectors for calibration.

No identity gain or SCIENTIFIC_SUCCESS is declared from the tiny smoke.

## Full screening and continuing recovery

Third commit `0a5ae2df` and fourth `3b5e6da3` pushed. Full clean screen has77 valid shared-ID queries/472 candidate pairs. GMT initial-pair logits and ReID cosine both R@1=100%; best VGGT symmetric layer17/ROIAlign R@1=74.026%, independent DINOv2 best center52% approximately. No GMT errors repaired on this easy fixed subset.74/232 detected targets cover no patch center; camera-order cosine change up to0.5738. Results are frozen exploratory evidence, not a tracking gain. A15-pair association-TRAIN feature extraction from videos5/8/12 fits scalar balanced logistic calibration (C=1 fixed), layer17 symmetric ROIAlign: slope3.0678, intercept-1.9661. No lightweight head is trained until independent complementary errors justify it.

The60-group feature residual integration smoke runs without crashes or fallback:6 update groups,12 actual VGGT calls. A separate other-user job appeared on GPU3 (PID39533,17GiB); its contention explains why this shared-GPU smoke cannot provide a fair speed claim. Our subsequent full GPU3 job (own PID13954) was terminated before producing completed predictions and restarted on idle GPU5 under tag `B2_l17_sym_w025_full_gpu5`. Other task untouched. Resource occupancy is not assumed static.

Masked/unmasked joint VGGT camera geometry on30 clean pairs:0/60 passes the preregistered independent-background validity gate. All outputs are neutral. This is evidence against the first camera-geometry route, not a project stop. Classical SIFT/ORB comparison, plane correspondence, conservative temporal alternatives and confidence diagnostics remain branches. First full masked-SIFT w0.5/k10 diagnostics on football14/basketball22 are unchanged in tracking metrics; no positive result declared.

Fresh B0 complete clean square10: HOTA80.9659, AssA81.0414, IDF1/CVIDF198.9676, IDSW0. Park18 predictions complete; strict evaluation follows. Real hard candidates are now collected on complete park18, selected by current normalized GTA ambiguity only, with no identity labels read by inference. This addresses the zero-hard-query problem in the initial screen. Read-only audit fields must pass prediction parity against B0 before any associated comparison.

## Stronger equivalence and actual causal entry

New geometry computation explicitly preserves Python, NumPy, CPU Torch and active CUDA RNG. Pure GMT's `_forward_asso` calls `jitter_bboxes_center` during inference, so a module must not disturb its RNG, even while detector boxes remain fixed. Actual VGGT compute-neutral control (6 updates/12 calls) exactly reproduces all298 pure-GMT smoke predictions. Complete park18 hard-event capture reproduces all11,671 B0 predictions exactly while collecting12 ambiguity events without GT selection. Original jitter, Hungarian and memory remain unchanged.

Added `gtr/geometry/online_runner.py` and `geo_tools/run_online.py`: one current synchronized group is read and processed, identities emitted immediately, no short-track deletion or delayed classification using future observations. Tracker uses original initial/GTA/memory functions; box fields are cloned before output postprocess so scaling cannot corrupt mutable history. Offline baseline is preserved. Actual input-to-identity latency includes current reading, preprocessing, inference, and CPU identity serialization; static model setup excluded and stated, cold lazy geometry initialization included. Prefix60 causal OFF runs; prefix80 comparison is underway. Causal enhanced and full-video accuracy remain outstanding.

First complete feature-residual video01 on GPU5 finishes; strict paired evaluation underway. This initial full run predates explicit RNG preservation and must be confirmed before interpreting any improvement.

## 2026-10-14 continuation: full first fusion and causal prefix verified

Previous goal turn is classified as progress: code, actual paired inference, neutral RNG control and causal entry were committed at `e58279095ef1a87f3fa98602e5d0c612f212599c`. Live process audit now confirms the first complete B2 layer17 symmetric weight0.25 / interval10 run has terminated successfully for all [1,10,18]. Fresh strict evaluation shows the same predictions/metrics as B0 on all three videos; this is a verified zero-change result, not a gain. No first-failure stop.

Causal OFF prefix60 versus prefix80: exact equality of the first300 serialized predictions, saved in results/causal_prefix_parity.json. Whole-video causal B0 park18 started; original offline and causal protocols remain separate.

The previous hard-event sampler incorrectly activated all proposal columns together; native global GMT activates each historical-frame block separately before GTA. Those old samples retain exact prediction parity but do not substantiate the claimed native GTA ambiguity. Corrected collection now uses the native per-frame unmatched softmax and keeps only target features for selected events. Full park18 collection under `hard_events_gta_v2` is running. This fixes sampling semantics before using any events for method selection or learning.

Remaining strict B0 scenes11 and19 started under `B0_strict_remaining_v1`. Independent DINOv2 TRAIN feature calibration and its fresh full-video residual control are being implemented; use the same frozen GMT, ROI mapping, training labels and scalar calibration recipe. This completes the tracking-level generic visual control that initial retrieval screening alone cannot substitute for.

Provenance now snapshots process-start commit as well as source file/diff hashes; report-generation commit is recorded separately, so commits made during long experiments cannot relabel their source.
