# Formal H=8 `video_06` shard audit — 2026-10-06

`video_06` reached the official COMPLETE boundary. This is a shard-level
integrity audit; it does not imply that the full H=8 dataset is complete.

## Results

- Records JSONL: 5,164 lines.
- Manifest `records`: 5,164.
- Manifest `source_main_decisions`: 5,164.
- Contract-invalid records: 0.
- Question counts: 2,600 `MATCH_DECISION`, 2,564 `MEMORY_DECISION`.
- Event-order range: 128,068–133,231.
- JSONL SHA256 matches the manifest:
  `sha256:9795987e0732482e42c6efe8c17ca7f282c153839d08899fde91652925c36cc5`.
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

The shard is therefore accepted as one official formal shard. The queue still
contains an incomplete failed `video_04` and many RUNNING shards; no full
dataset merge or policy training is authorized yet.
