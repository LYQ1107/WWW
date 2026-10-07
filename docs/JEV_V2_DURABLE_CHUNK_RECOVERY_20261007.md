# Durable JEV v2 chunk recovery

Status: implemented and CPU-tested on the future-development branch. This
protocol is for a future frozen-worktree canonical H=8 rebuild. It does not
modify or retrofit the currently running video01 v2 process.

## Boundary and ordering

`run_jev_intra_video_chunk.py` now supports `--progress-file`,
`--checkpoint-file`, and `--resume`.

After every completed production key, the worker performs this ordering:

1. append each emitted JSONL record and `fsync` it;
2. atomically replace `progress.json`;
3. atomically replace a Torch checkpoint containing the mutable GMT state,
   trajectory RNG state, progress, and exact input/source provenance.

The checkpoint is therefore only allowed to advance after all records for the
same key are durable. A crash during a key can leave extra records in the
partial JSONL; resume truncates only those records after the last durable
checkpoint and restarts from the checkpointed key boundary.

## Fail-closed resume

Resume requires an exact match for:

- canonical source commit;
- video/chunk and key range;
- trace and order-index hashes;
- partition and chunk-plan hashes;
- GMT checkpoint hash;
- horizon, backend, and state schema.

Any mismatch or invalid state signature aborts resume. The completed artifact
is still written as `records.jsonl` plus `manifest.json`, and the durable
recovery files remain available for audit.

## Verification

The following checks passed in the GMT environment:

```text
python -m py_compile build_jev_counterfactual_v2.py \
  jev_v2_progress_recovery.py run_jev_intra_video_chunk.py
pytest -q reproduction_tools/test_jev_v2_progress_recovery.py
2 passed
```

A live-data CPU smoke also ran the new `progress_callback` through one
cosine-contract record: one record emitted, one callback, three branch
expansions, and a non-null trajectory RNG seed. No GPU or formal artifact was
modified by this smoke test.

## Canonical-use restriction

Before using this runner for the 24-video canonical dataset, a single-worker
versus chunked semantic/byte-equivalence test must pass on a controlled video.
This branch is a recovery implementation, not authorization to resume the
paused speculative Full H8 build.
