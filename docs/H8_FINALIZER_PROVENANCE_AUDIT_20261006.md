# Formal H=8 finalizer and provenance audit — 2026-10-06

## Finalizer correction

The first complete shard exposed a representation mismatch at the merge boundary. Records contain the checkpoint digest as a bare hexadecimal string, while the shard manifest records it as `sha256:<digest>`. The finalizer compared those strings literally and would have rejected every valid shard.

`reproduction_tools/finalize_jev_full_h8.py` now normalizes only this optional display prefix before comparing the record and manifest. It does not change the digest, the records, the cache, or any training/evaluation semantics. The finalizer's strict `source_commit` consistency check remains unchanged.

## Evidence

The complete `video_08` shard was re-audited with the corrected boundary:

- 3,950 records in the manifest and JSONL;
- 1,997 `MATCH_DECISION` and 1,953 `MEMORY_DECISION` records;
- all 3,950 records pass `validate_record(..., allow_future_gt=True)`;
- all records are H=8 and match the manifest checkpoint after prefix normalization;
- event orders span 136,569–140,518;
- the JSONL SHA256 matches `records_artifact_sha256` in the manifest.

The fixture regression test `reproduction_tools/test_finalize_jev_full_h8.py` passes with a bare record digest and a prefixed manifest digest.

## Remaining provenance decision

`video_08` was produced under source commit `1bf00c95`. The current branch is `eee26d1`, which contains the empty-zero-detection ReID adapter fix needed for the failed `video_04` retry. The finalizer still requires all shards to have the same raw `source_commit`; this gate must not be silently weakened. Before the full merge is declared PASS, the retry source lineage must be explicitly resolved and recorded.

No active builder was stopped, no queue entry was changed, and `video_04` was not retried by this audit.
