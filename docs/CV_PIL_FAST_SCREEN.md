# CV-PIL fast screen — preregistration (2026-10-15)

**Status: REAL_CACHED_TRAIN_EXPERIMENT_PENDING.** No VisionTrack accuracy, HOTA, FPS or architecture gain is claimed until actual cached TRAIN measurements run on the user's server. This branch implements a compact screen and CPU synthetic checks, not a new GMT/JEV Master Goal.

## Historical evidence and hypothesis

- Complete frozen GMT+GTA with an ordinary residual MLP produced +0.600573 HOTA / +1.456462 CVIDF1 on a previously exposed TRAIN16 development video, at +47 IDSW and 8.21% lower full-image FPS. This is **not** CV-PIL evidence.
- The typed-JEV two-plan verifier lacked independent supervised decisions (eight certified APPLY, zero KEEP). This probe instead asks whether predicting a person's future appearance in another camera from strictly earlier observations contributes *beyond* an ordinary camera-conditioned metric learner.
- Existing [TRAIN cache audit](../results/history_rank_train_audit_v1.json) recorded TRAIN5/8/12 compact observations 160/151/595, GMT descriptors, detector-aligned events/native GTA logits, and frozen DINOv2/VGGT features. No new video download, encoder call, or dense feature cache is necessary.

## Primary real-data probe

[geo_tools/quick_cv_pil_native.py](../geo_tools/quick_cv_pil_native.py) uses the **original native GTA candidate event list**, not GT-constructed online candidate IDs. A reference must be strictly earlier than the query and come from another camera; a proposed source-frame cohort must contain one known positive and at least one known negative. GT is used offline as labels, never as model input.

Run in a worktree of this GitHub research branch:

    /home/liuyeqiang/anaconda3/envs/GMT/bin/python geo_tools/quick_cv_pil_probe.py --self-test
    /home/liuyeqiang/anaconda3/envs/GMT/bin/python -m unittest geo_tools/test_quick_cv_pil_probe.py geo_tools/test_quick_cv_pil_native.py
    /home/liuyeqiang/anaconda3/envs/GMT/bin/python geo_tools/quick_cv_pil_native.py --runtime /data1/liuyeqiang/WWW_vggt_geo_runtime --out /home/liuyeqiang/WWW_cvpil_native_20261015 --steps 300 --device auto

The output path must not exist. Refuse to regenerate missing caches, verify manifest and SHA, preserve >30GiB free disk on the destination, and leave data/checkpoints untouched. Actual branch/commit/retained metrics must be recorded after running.

## Locked methods (TRAIN5+TRAIN8 training, TRAIN12 diagnosis)

- N0 original native GTA proposal logits (no trained head)
- N1 native GTA logits + 2 × frozen GMT VFCE1024 descriptor cosine
- N2 same native fusion with rank16 **linear** camera/age-conditioned projection
- N3 same native fusion with rank16 nonlinear projection and ranking-only objective
- N4 **CV-PIL:** identical N3 network, sequence and ranking loss, plus a latent prediction loss for future opposite-camera query appearance
- N5 N4 with destination camera ID removed
- N6 **optional cached DINOv2 teacher distillation:** N4 augmented by KL to teacher candidate ranking, only if teacher beats native on TRAIN5/8 *without examining TRAIN12*; no DINOv3/V-JEPA heavyweight inference

Use seed 20261015, 300 fixed steps per trained arm, rank16 bottleneck, residual scale0.25, native fusion scale2, camera/time conditions. N3/N4 use exactly the same sampled episodes; no train/validation mixing or hyperparameter selection on TRAIN12.

Store eligible clause counts and **unique query/person** counts, past-frame causality checks, original cache SHA, native/descriptor/each head Recall@1 and MRR, paired fixes/harms, and final losses.

**Go gate:** N4 or eligible N6 exceeds N0/N1/N2/N3 on identical TRAIN12 candidate cohorts, with native paired fixes > harms. Even then status is only WORTH_NATIVE_CLOSED_LOOP_REVIEW, **not** an online tracking success. If negative, stop and do not start expensive end-to-end tracking.

## Secondary oracle-gallery control

[geo_tools/quick_cv_pil_probe.py](../geo_tools/quick_cv_pil_probe.py) additionally offers a GT-grouped historical gallery. That is explicitly **offline oracle-gallery retrieval only**. It must never be reported as deployable native candidate generation or whole-video association. Do not select hyperparameters or claim online improvements from it.

## Evidence limitations

Native-event screening fixes the original tracker candidate lists and learns only candidate representation scores. It does *not* execute full Hungarian, births, Bank, gallery mutation, HOTA, CVIDF1, IDSW, real FPS, or blind generalization. TRAIN12 and all legacy backbones had historical training exposure. Dependent native past-frame clauses are not independent persons or repeated seeds. Frozen DINOv2 optional teacher does not demonstrate 2026 foundation-model novelty.

If a native screen succeeds, next stage must verify a complete frozen GMT+GTA integration, alongside the existing GMT+MLP and a matched plain learner, on properly authorized full native video. Retain negative VGGT/JEV conclusions. Do not open sealed videos20/21/22, new DEV17–19, official TEST or unauthorized Full24.

**Assistant-side validation performed:** one synthetic CPU end-to-end diagnostic (four 300-step trained arms) and five distinct synthetic unittest cases. Synthetic data have no VisionTrack scientific meaning. The assistant cannot access the user's GPU workstation or private feature cache directly from the connected GitHub source, so REAL TRAIN screening remains NOT_RUN.
