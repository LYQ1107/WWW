# Identity memory

`WorldIdentityMemory` uses world-center distance, covariance-aware gating, normalized query features, and a deterministic Hungarian assignment. It allocates monotonically increasing IDs, confirms after repeated hits, predicts short gaps, marks long gaps LOST, and deletes after the fixed missed-frame limit. Scene reset clears all state.

This is an explainable baseline inspired by the lifecycle concerns in SparseQID. SparseQID checkpoints and detector code are not loaded.
