# WorldMOT v2 architecture

The causal path is:

```text
two synchronized RGB images
        ├── frozen RTMDet-Ins-tiny → bbox/mask/score
        └── frozen VGGT-Ω 512       → depth/pose/pointmap/patch tokens
                         ↓
       independent MultiViewObservationEncoder tokens
                         ↓
       PersistentWorldQueryTransformer (Q(t-1) → Q(t))
                         ↓
       assignment / birth / visibility / world-state heads
                         ↓
       camera-conditioned diagnostic projection and memory(t+1)
```

The encoder never calls the v1 `fuse_two_views()` path.  `camera_id` and
camera-conditioned geometry are retained in each token and enter a learned
camera embedding, so a query can read
two views jointly while the final runtime constraint enforces one observation
per query per camera and one query per observation.

`slot_id` is a decoder index.  `global_id` is allocated by
`QueryMemoryBank`; slot reuse therefore cannot silently reuse an old identity.
The current query xyz head is normalized to a clip coordinate convention and
is not treated as metric world geometry.

