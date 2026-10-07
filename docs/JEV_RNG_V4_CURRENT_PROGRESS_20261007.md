# JEV RNG-isolation v4 current progress

Snapshot: `2026-10-07T00:59Z`  
Branch: `jev/counterfactual-rng-isolation-v4-20261006`

This is a timestamped progress snapshot, not a final research result.

## Verified completed gates

- Corrected video06 runtime feature parity: `PASS`, `5162/5162` records,
  three repetitions, zero missing records, zero OFF-action mismatches.
- Corrected video07 runtime feature parity: `PASS`, `3334/3334` records,
  three repetitions, zero missing records, zero OFF-action mismatches.
- Frozen numerical policy: absolute `2e-5` for bounded/state features plus
  relative `4 * eps32 = 4.76837158203125e-7` only for the explicitly listed
  unbounded float32 score-derived features. The stricter `1e-7` diagnostic
  failure is preserved separately.
- Same-code formal video07 single-worker/three-chunk equivalence: `PASS`,
  `3337/3337` semantic keys, no range gaps/overlaps, exact canonical SHA and
  exact RNG provenance. Full-H8 chunking is therefore authorized.
- Corrected canonical-feature v6/v7 compact dataset and single-seed training:
  `8501` records, H=8, sequence-disjoint split, seed `20261003`, Threshold /
  Generic MLP / Full JEV checkpoints and calibration artifacts present.
  The current offline report is a screening comparison; it does not by itself
  establish a meaningful JEV advantage over the majority-action reference.

Evidence:

- `reports/JEV_RNG_V4/RUNTIME_FEATURE_PARITY_VIDEO06_CURRENT_HEAD.json`
- `reports/JEV_RNG_V4/RUNTIME_FEATURE_PARITY_VIDEO07_CURRENT_HEAD_TOL2E5.json`
- `reports/JEV_RNG_V4/FORMAL_GMT_INTRA_VIDEO_CHUNK_EQUIVALENCE_VIDEO07_CURRENT_HEAD_V10.json`
- `reports/JEV_RNG_V4/CORRECTED_V4_SMALL_H8_THREE_WAY_VIDEO06_VIDEO07.json`

## Live work at snapshot time

- Corrected video01 builder PID `8251` remains active on GPU0 and has not been
  stopped or migrated. Its completed artifact is required before the final
  three-question runtime gates can run.
- The Full H=8 scheduler is `RUNNING` with workers on the authorized safe GPUs
  for video15, video16, video23, and video24. The remaining videos stay queued.
  No frozen GMT baseline or full VISION_test OFF baseline is being rerun.
- The video01 aftercare and corrected tracking waiters are running in a
  fail-closed state and consume no GPU while waiting for the video01 manifest.

## Required next order

1. Validate the completed corrected video01 artifact.
2. Run full MATCH/MEMORY/REACTIVATION runtime feature parity.
3. Run native-vs-replay reactivation candidate parity and repeated numerical
   stability checks.
4. Run the corrected-v4 closed-loop GMT OFF / Threshold / MLP / JEV comparison
   using the canonical corrected-feature controllers.
5. After all 24 Full-H8 manifests complete, run the audited finalizer, compact
   the full dataset, create the TRAIN-only sequence split, and publish the
   final single-seed results. Multi-seed robustness remains deferred until the
   first-round gate is meaningful.

No final tracking claim is made until steps 1--4 pass and the resulting report
is pushed to this branch.
