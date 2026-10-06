# Formal H=8 `video_07` shard audit — 2026-10-06

`video_07` reached the official COMPLETE boundary. This is a shard-level
integrity audit; the full formal dataset is still incomplete.

## Results

- Records JSONL: 3,337 lines.
- Manifest `records`: 3,337.
- Manifest `source_main_decisions`: 3,337.
- Contract-invalid records: 0.
- Question counts: 1,703 `MATCH_DECISION`, 1,634 `MEMORY_DECISION`.
- Event-order range: 133,232–136,568.
- JSONL SHA256 matches the manifest:
  `sha256:83d741bdf5a633f0f249f4e68db8a8963fae0dfb410a827dfe3971f461006e37`.
- Checkpoint: canonical Stage2 `model_20000.pth`,
  `sha256:cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8`.

## Provenance

- Source commit: `1bf00c95cc07c6037bd5df1767f64fb6d912d1a5`.
- Association backend: `formal_gmt_transformer`.
- Counterfactual engine: `cached_perception_mutable_association_v2`.
- State schema: 2.
- Horizon: H=8.
- Sampling and truncation: both false.
- Cache, trace, checkpoint, and annotation provenance remain bound to the
  formal shard manifest.

The shard is accepted as one official formal shard. The queue still contains
the failed `video_04` and many RUNNING shards; no full merge or policy
training is authorized yet.
