# Formal H=8 partial-artifact audit — 2026-10-06 08:15 UTC

This is a read-only audit of the temporary per-video JSONL artifacts while
the formal builders remain active. It does not modify queue state, stop a
worker, or promote any temporary file to an official shard.

## Result

- 24 partition videos were scanned.
- 62 temporary artifacts were found.
- All 62 artifacts passed JSON/schema checks (`bad_records=0`).
- No artifact reached its video's full expected-record boundary without a
  corresponding COMPLETE manifest.
- Therefore `recovered_complete_videos=[]`; no temporary artifact is eligible
  for automatic reuse or merge.

The audit checks the frozen H=8 contract fields, including `schema_version=1`,
`horizon=8`, the canonical Stage2 checkpoint digest,
`formal_gmt_transformer`, video identity, event order, legal actions, and
horizon outcomes. An artifact can still be an incomplete live prefix even
when every line currently on disk is structurally valid.

`video_04` remains a failed, non-official 4,937-record prefix out of 7,203
expected records. Its valid lines are retained as diagnostic evidence only;
they are not reused automatically.

The machine-readable evidence is in
`reports/H8_FORMAL_PARTIAL_REUSE_AUDIT_20261006_0815.json`.
