# WorldMOT World Object Fast Proof — 2026-10-25

This is an independent experiment scope. It does not resume the stopped V2
multi-seed queue and it does not modify `WWW/main` or any frozen GMT result.

## Frozen scope

- Branch: `research/worldmot-worldobject-fast-proof-20261025`
- Base: verified GMT/VGGT point-comparison source (`660054773a0492fb3f32c1f42587a04619cad5b6`)
- Dataset: VisionTrack, TRAIN 24 scenes / 28,754 synchronized timestamps;
  TEST 22 scenes / 29,019 synchronized timestamps.
- Geometry: verified VGGT-Ω 512 source/checkpoint, full synchronized RGB pair.
- Detector: immutable GMT pre-GTA candidate cache; no identity IDs or GT are
  available to inference.
- NAS root: `/data1/liuyeqiang/nas/A100/WorldMOT/worldobject_fast_proof_20261025/`.

The source/data/resource lock is saved as
`diagnostics/scope_lock.json`; the actual dataset audit is
`diagnostics/data_audit.json`. Cache shards are atomic, checksum-verified and
claimed by an atomic scene queue. The six cache slots used physical GPUs
2/3/4/6/7/8; unrelated user processes on other GPUs were left untouched.

## Architecture under test

The new route keeps a persistent world object as the association unit:

1. frozen current-frame GMT candidates;
2. frozen Ω pointmaps and causal dual-camera static alignment;
3. box support plus a separate depth-cluster foreground support proposal;
4. joint multi-view world-object hypotheses (one detection/object and at most
   one detection/camera/object);
5. `PersistentWorldObjectBank` lifecycle and current-detection-only output.

The foreground proposal is not called an instance mask. If depth is not
separated, the ambiguity code is retained and the object is allowed to abstain
or fall back conservatively.

All detector boxes are kept in the original RGB coordinate system for output,
but are explicitly transformed through Ω's registered crop/resize/pad
`image_to_model` matrix before pointmap support sampling. A pre-fix partial
cache is retained under `diagnostics/` and is not eligible for any metric.

## Registered routes

- G0-A: original first-camera Sim(3) alignment.
- G0-B: dual-camera static-background Sim(3) alignment.
- G0-C: causal keyframe-anchored alignment (engineering route; evaluated only
  after TRAIN cache verification).
- G1-A/B/C: rectangle lift, foreground-aware support, compact support.
- G2-A/B/C: pairwise control, robust support fusion, joint object formation.
- G3-A/B/C: original memory, persistent object bank, oracle-vs-predicted
  diagnostic separation.
- W4: small persistent world-object update head, frozen GMT/Ω.
- W5: no-geometry capacity control and detection-preserving control.

## Evaluation guard

The official TEST score is valid only when all 22 videos have complete causal
prediction files. The original raw GT is never modified; the eight GT-invalid
scenes remain inference/coverage-only and the original legal scenes are the
only official scoring scope. Cache replay FPS is reported separately from live
RGB-to-ID E2E FPS.

No new multi-seed score is registered in this scope. A seed is not used to
select a TEST threshold or architecture.

## Current status

At the initial engineering checkpoint, the data audit and Ω support-cache
smoke passed, the four-card TRAIN cache scheduler was running, and the W0–W3
auditors, W4 training entry point and TEST replay entry point passed compile and
core unit tests. Final numerical conclusions are written only after the full
24-scene TRAIN gate and full 22-video TEST coverage gate.

