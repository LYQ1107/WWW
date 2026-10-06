"""Create a non-destructive partition manifest for completed H=8 videos.

The canonical partition is never edited.  This helper only narrows its
manifest and cache-key index so completed videos can enter the small-gate
finalizer while another video is still being built.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-root", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--video-ids", type=int, nargs="+", required=True)
    args = parser.parse_args()

    source_root = args.partition_root.resolve()
    shard_root = args.shard_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output_root}")
    source_manifest = json.loads(
        (source_root / "partition_manifest.json").read_text(encoding="utf-8")
    )
    if source_manifest.get("status") != "COMPLETE":
        raise ValueError("source partition manifest is not COMPLETE")
    source_videos = source_manifest.get("videos") or {}
    selected = sorted({int(value) for value in args.video_ids})
    if not selected:
        raise ValueError("at least one video is required")
    cache_index = json.loads(
        (source_root / "cache_keys_by_video.json").read_text(encoding="utf-8")
    )
    videos: dict[str, dict[str, Any]] = {}
    cache_keys: dict[str, Any] = {}
    for video_id in selected:
        key = str(video_id)
        if key not in source_videos or key not in cache_index:
            raise KeyError(f"video {video_id} is absent from the source partition")
        shard = shard_root / f"video_{video_id:02d}"
        manifest_path = shard / "manifest.json"
        records_path = shard / "records.jsonl"
        if not manifest_path.is_file() or not records_path.is_file():
            raise FileNotFoundError(f"video {video_id} is not an official shard: {shard}")
        shard_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if shard_manifest.get("status") != "COMPLETE":
            raise ValueError(f"video {video_id} shard is not COMPLETE")
        expected = int(source_videos[key]["main_decisions"])
        if int(shard_manifest.get("records", -1)) != expected:
            raise ValueError(
                f"video {video_id} records mismatch: "
                f"{shard_manifest.get('records')} != {expected}"
            )
        videos[key] = dict(source_videos[key])
        cache_keys[key] = cache_index[key]

    stats = dict(source_manifest.get("stats") or {})
    stats["video_count"] = len(selected)
    stats["nonempty_lines"] = sum(int(v["lines"]) for v in videos.values())
    stats["main_decisions"] = sum(int(v["main_decisions"]) for v in videos.values())
    stats["legal_action_count"] = sum(int(v.get("legal_action_count", 0)) for v in videos.values())
    question_counts: dict[str, int] = {}
    for video in videos.values():
        for question, count in (video.get("question_counts") or {}).items():
            question_counts[question] = question_counts.get(question, 0) + int(count)
    stats["question_counts"] = question_counts
    stats["total_partition_lines"] = stats["nonempty_lines"]
    stats["total_main_decisions"] = stats["main_decisions"]
    manifest = dict(source_manifest)
    manifest.update(
        {
            "status": "COMPLETE",
            "schema_version": "jev_rng_v4_completed_subset_v1",
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source_partition_root": str(source_root),
            "source_partition_manifest_status": source_manifest.get("status"),
            "selected_video_ids": selected,
            "videos": videos,
            "stats": stats,
            "total_partition_lines": stats["total_partition_lines"],
            "total_main_decisions": stats["total_main_decisions"],
            "canonical_authority": "SMALL_GATE_PARTIAL_NOT_FULL_CANONICAL_H8",
            "selection_reason": "completed_shards_only_while_video1_continues",
        }
    )
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "partition_manifest.json", manifest)
    write_json(output_root / "cache_keys_by_video.json", cache_keys)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "output_root": str(output_root),
                "video_ids": selected,
                "total_main_decisions": stats["total_main_decisions"],
                "question_counts": question_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
