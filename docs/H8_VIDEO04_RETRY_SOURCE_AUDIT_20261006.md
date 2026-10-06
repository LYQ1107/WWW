# H=8 `video_04` retry source audit — prepared, not authorized

This report records a retry source prepared in an isolated worktree. It is
not a retry result, does not change the formal queue, and does not authorize
mixing the retry shard into the merged dataset.

## Source lineage

- Existing COMPLETE formal shard lineage: `1bf00c95cc07c6037bd5df1767f64fb6d912d1a5`.
- Isolated retry branch: `jev/h8-video04-retry-source`.
- Isolated retry commit: `905a5dde15139ebb1fadbcabed6916b54160a150`.
- The isolated worktree was based directly on the existing formal source
  commit; it was not based on the moving research branch.

The retry commit changes only the GMT association adapter and its regression
test relative to the base commit. The adapter implementation is byte-identical
to the already validated empty-ReID-frame fix on the main research branch:

`ad2969497b1226988966121778778b6bf256eeb4288d1bc1b0fd6c84299da0a0`

## Validation completed

- Python compilation: PASS.
- GMT association-adapter invariant test, including empty historical/current
  frames: PASS.
- Retry branch pushed to GitHub: PASS.

## Gates still required

The following are intentionally still `PENDING`:

1. A bounded builder/equivalence check against the existing formal source
   lineage, including exact record count, state, legal actions, best action,
   and declared utility tolerance.
2. A safe GPU slot while all currently valid builders remain untouched.
3. A complete `video_04` manifest, JSONL hash, schema audit, and source-order
   prefix audit.
4. An explicit `PASS` compatibility manifest before invoking the opt-in mixed
   source finalizer gate.

Until all four gates pass, the failed 4,937-record prefix remains evidence
only and must not be promoted, merged, or used for policy training.
