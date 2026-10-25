# WorldObject V3: joint birth repair and full-video verification

Branch: `research/worldmot-v3-birthfix-sixgpu-20261025`.
Base: `c2c9c7f7ee03b6dfa160f23b2f44d2e7339b635d`.
User instruction: attachment `c5d7a93f-e5e6-4068-83b2-45635a8dd642`.

## Status

Phase A implementation and phase B CPU acceptance are complete. Real TRAIN preflight,
six-GPU work, full TEST and official comparison are pending. No new accuracy result
is claimed in this revision. This is an independent WorldMOT experiment; GMT is
a frozen detector/ROI provider and a separately scored reference.

## Repairs

- A chunked, trainable current-view pair head supplies symmetric joint-birth
  evidence to the actual constrained assignment solver. Its clean new-identity
  positives and different-identity negatives are distinct from the auxiliary
  current-view identity loss. Unmatched and ambiguous TRAIN labels are ignored.
  Candidate generation does not use GT. Logs record model/proposal retention.
- Predicted history is continuous within each scene. The 64-step boundary is
  logging/graph truncation only. Duplicate historical objects with the same TRAIN
  identity produce multi-positive historical targets instead of false births.
  Teacher history is a separate, after-forward TRAIN diagnostic.
- Point availability, alignment validity, reliability, fusion eligibility and
  world-state update eligibility are separate fields. Weak depth support cannot
  force spatial fusion or overwrite a reliable spatial state. Current detections
  remain available to the visual fallback.
- The capacity-matched appearance name is a deprecated alias of full no-geometry,
  with no duplicate training job. The no-learned-birth experiment reuses the full
  trained checkpoint with birth-pair decision evidence disabled.
- Completed cache scenes are checksum/provenance verified and reused. Partial
  scenes are preserved in a recovery directory and restarted at the scene origin.
  Atomic scene claims prevent duplicate six-worker reconstruction. A concurrent
  claim/read race was found by CPU testing and corrected.
- Native MATLAB progress is saved per job, and TrackEval output is saved before
  MATLAB. A timeout terminates only the evaluator's own process group. Native
  scoring routines and original GT are unchanged.

## Phase B evidence

`results/worldobject_v3_birthfix_cpu_test_report.json` records 104 passing CPU tests,
including the original acceptance suite. The actual Ω Python environment was used:
Torch 2.6.0+cu118, with CUDA hidden. Changed Python files also pass Pyflakes.
These are engineering tests, not real-video MOT measurements.

## Frozen experiment registration

`configs/worldobject_v3_birthfix_registered.json` fixes seed 20261025, hidden size
128, the 18/6 TRAIN split, four selection epochs and the TRAIN-only extension rule
(maximum six), all-24-scene reinitialization, continuous predicted history,
joint-birth/auxiliary loss weights, geometry eligibility thresholds and candidate
limits. Full TEST contains 22 scenes; only the original 14 legally evaluable
scenes enter official aggregates. Eight blocked scenes still receive inference.
No TEST outcome may select a checkpoint or change thresholds.

Single-card engineering preflight is restricted to TRAIN scenes 1, 3, 8 and 24
(130, 16, 16 and 16 paired timestamps). Six-card execution is gated on this
preflight, checks physical GPU occupancy and binds UUIDs. Per-video latency is
reported separately from parallel aggregate throughput. Full no-geometry may
reuse immutable ROI-restored object caches; geometry methods require the new
camera-depth/persistent-alignment cache contract.

## Results awaiting execution

Training coverage, full TEST coverage, official metric comparison, birth/error
attribution, and live runtime breakdown will be added as real jobs finish.
The prior V3 results are preserved and are not results of this repaired model.
