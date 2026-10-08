# Phase VIII storage cleanup

P-1 is complete. Examined 656 model/checkpoint/state-snapshot files (37.764 GiB) in explicitly scoped project directories. SHA256 values and verified content duplicates are in CHECKPOINT_INVENTORY.csv. Indexed per-video/frame/view perception `.pt` tensors are protected data, inventoried separately rather than misclassified as model checkpoints.

Deleted files: none. Actual released space: **0 bytes / 0 GiB**. No artifact met all ten verified-orphan conditions; missing original training provenance and intentionally omitted large raw reference files cannot prove absence of a dependency. Classes: {"A_PROTECTED": 381, "B_REPRODUCIBILITY_REQUIRED": 104, "D_UNKNOWN": 171}. Negative experiments and all Phase V/VI/VII evidence remain. Archived historical training weights, the permanent B2, GMT foundation, fork snapshots and unidentified files are retained.

Default dry run and explicit empty execution both completed. Eight adversarial cleanup guard tests pass. No other user's process, system cache or Git history was altered. Free-space changes made by other tasks are not credited as cleanup.

/data1 has substantial space pressure; /home is a separate filesystem with roughly 99 GB available at the initial audit. New runtime outputs will use /home, reuse the frozen perception cache and keep bounded forks plus best/last checkpoints. A sparse isolated worktree will avoid copying old bulky research archives. GitHub publication remains limited to source, configuration, metrics, compact examples and SHA/provenance manifests.

WHAT DID WE LEARN? A failed or inactive run is not a safe deletion proof. Storage safety can be maintained with zero deletion through output placement, cache reuse and smaller publication scope.
