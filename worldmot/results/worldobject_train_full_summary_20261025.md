# World Object full TRAIN gate (2026-10-25)

The shared frozen cache completed before either model was trained:

- 24/24 TRAIN scenes, 28,754/28,754 synchronized timestamps.
- Cache schema: `worldobject_omega512_sparse_support_v1`.
- One source SHA and one Ω checkpoint SHA across all manifests; no TEST data was read by training.
- Validation scenes: 4, 8, 12, 16, 20, 24. Fit scenes: the other 18. Final checkpoint was then reinitialized with the same seed and retrained on all 24 scenes.
- Ambiguous or missing identity labels were explicitly masked and counted; raw labels and cache rows were not modified.

| Route | TRAIN status | Parameters | Checkpoint SHA-256 | Geometry input |
|---|---|---:|---|---|
| W4 learned update | `FULL_TRAIN_GATE_PASSED` | 23,186 | `ce8ca23a1665e7f3b97eba6fcf365aafe5d93f1c5213f243e3d6e0d31c066317` | persistent world-object geometry |
| W5 capacity control | `FULL_TRAIN_GATE_PASSED` | 23,186 | `00313ebeca5286703ed4af625767f621a81a604f3e89bc14dcab31578b0141da` | zeroed geometry |

Both reports record final-scene coverage of 24 scenes and 28,754 timestamps. W5 is a capacity control, not a claim that geometry is unnecessary; its TEST score must be compared under the same replay and evaluation protocol.

The full per-scene reports and checkpoints remain in the NAS experiment directory; large binary artifacts are intentionally not uploaded to GitHub.
