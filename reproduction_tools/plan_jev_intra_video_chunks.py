"""Create a deterministic future-canonical intra-video chunk plan."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

from build_jev_counterfactual_v2 import event_maps
from build_jev_counterfactual_dataset import normalize_events
from jev_intra_video_chunking import CHUNKING_SCHEMA_VERSION, plan_chunks


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--video-id", type=int, required=True)
    parser.add_argument("--records-per-chunk", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    partition = json.loads(args.partition_manifest.read_text(encoding="utf-8"))
    video_id = int(args.video_id)
    video_meta = partition["videos"][str(video_id)]
    trace_path = Path(partition["trace_by_video"]) / f"video_{video_id:02d}.jsonl"
    order_path = Path(partition["trace_by_video"]) / f"video_{video_id:02d}.orders.jsonl"
    events = normalize_events(trace_path, order_index=order_path, minimal=True)[video_id]
    _actions, _memories, by_key = event_maps(events)
    cache_keys_payload = json.loads(
        (Path(args.partition_manifest).parent / "cache_keys_by_video.json").read_text(encoding="utf-8")
    )
    keys = cache_keys_payload[str(video_id)]
    chunks = plan_chunks(
        video_id=video_id,
        ordered_keys=keys,
        events_by_key=by_key,
        target_records=args.records_per_chunk,
    )
    report = {
        "schema_version": CHUNKING_SCHEMA_VERSION,
        "status": "PASS",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "FUTURE_CANONICAL_H8_ONLY_NOT_ACTIVE_SMALL_GATE",
        "video_id": video_id,
        "trace": str(trace_path),
        "trace_sha256": sha256(trace_path),
        "order_index": str(order_path),
        "order_index_sha256": sha256(order_path),
        "partition_manifest": str(args.partition_manifest.resolve()),
        "partition_manifest_sha256": sha256(args.partition_manifest.resolve()),
        "cache_keys": len(keys),
        "source_main_decisions": int(video_meta["main_decisions"]),
        "records_per_chunk_target": int(args.records_per_chunk),
        "chunk_boundary_unit": "complete ordered (video_id, frame, view) cache-key decision unit",
        "chunks": chunks,
        "chunk_count": len(chunks),
        "total_planned_decisions": sum(item["decision_count"] for item in chunks),
        "warmup_policy": "one production-order GMT-OFF pass; snapshot MutableGMTState and explicit trajectory RNG at each chunk start",
        "equivalence_policy": "single-worker vs merged chunked build must match exact canonical records by semantic key",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "video_id": video_id, "chunks": len(chunks), "decisions": report["total_planned_decisions"]}, indent=2))


if __name__ == "__main__":
    main()
