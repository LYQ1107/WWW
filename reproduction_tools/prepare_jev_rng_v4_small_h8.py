"""Prepare a non-destructive 2--3-video H=8 RNG-controlled build manifest."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-partition", type=Path, required=True)
    parser.add_argument("--output-partition", type=Path, required=True)
    parser.add_argument("--video-ids", type=int, nargs="+", required=True)
    args = parser.parse_args()
    source_root = args.source_partition.resolve()
    output_root = args.output_partition.resolve()
    source_manifest_path = source_root / "partition_manifest.json"
    source_keys_path = source_root / "cache_keys_by_video.json"
    if not source_manifest_path.is_file() or not source_keys_path.is_file():
        raise FileNotFoundError("source partition is missing its manifest or cache keys")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "COMPLETE":
        raise RuntimeError("source partition is not COMPLETE")
    video_ids = sorted({int(value) for value in args.video_ids})
    if not video_ids:
        raise ValueError("at least one video is required")
    missing = [str(value) for value in video_ids if str(value) not in source_manifest["videos"]]
    if missing:
        raise ValueError(f"video ids are absent from source partition: {missing}")
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output partition: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    selected_videos = {
        str(video_id): source_manifest["videos"][str(video_id)]
        for video_id in video_ids
    }
    manifest = dict(source_manifest)
    subset_stats = dict(source_manifest.get("stats", {}))
    subset_stats["video_count"] = len(selected_videos)
    subset_stats["main_decisions"] = sum(
        int(item["main_decisions"]) for item in selected_videos.values()
    )
    subset_stats["nonempty_lines"] = sum(
        int(item["lines"]) for item in selected_videos.values()
    )
    question_counts = {}
    for item in selected_videos.values():
        for question, count in item.get("question_counts", {}).items():
            question_counts[question] = question_counts.get(question, 0) + int(count)
    subset_stats["question_counts"] = question_counts
    manifest.update(
        {
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "videos": selected_videos,
            "stats": subset_stats,
            "total_main_decisions": sum(
                int(item["main_decisions"]) for item in selected_videos.values()
            ),
            "total_partition_lines": sum(
                int(item["lines"]) for item in selected_videos.values()
            ),
            "small_h8_selection": {
                "video_ids": video_ids,
                "all_decisions": True,
                "sampling": False,
                "truncation": False,
                "purpose": "RNG_FIXED_GATE_BEFORE_CANONICAL_REBUILD",
            },
            "source_partition_manifest_sha256": "sha256:" + sha256(source_manifest_path),
        }
    )
    (output_root / "partition_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copy2(source_keys_path, output_root / "cache_keys_by_video.json")
    print(json.dumps(
        {
            "status": "PASS",
            "output_partition": str(output_root),
            "video_ids": video_ids,
            "main_decisions": manifest["total_main_decisions"],
            "partition_manifest_sha256": "sha256:" + sha256(output_root / "partition_manifest.json"),
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
