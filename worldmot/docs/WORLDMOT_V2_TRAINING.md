# Training contract

`WorldQueryClipDataset` consumes object-level feature-cache shards.  A shard
must contain `observations`, `geometry`, `assignment`, `birth`, and
`visibility`; it is rejected if it contains oracle xyz fields.  The finite
training entry point is `scripts/train_world_query.py`.  It uses AdamW,
explicit seeds, checkpoint manifests and a real backward pass.  Ω and the
detector remain frozen feature sources in this stage.

The current local manifest records a three-frame Scene 8 real-feature pilot
with a finite backward pass; its checkpoint is outside Git and is not a formal
MOT model.  Full 24-scene TRAIN, multi-seed validation, and a calibrated
world-coordinate projection head remain required before a formal MOT claim.

