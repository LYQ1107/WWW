"""Partition the canonical TRAIN trace for restart-safe full H=8 replay.

The partitioner copies raw JSONL lines without rewriting them.  Each per-video
partition has a sidecar containing the original source line number so replay
can preserve ``online_context.event_order`` exactly even though workers only
read their own video.  This is an I/O preparation step; it does not sample,
truncate, relabel, or otherwise transform the decision data.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Dict, Mapping

from jev_mutable_branch import is_main_decision


SCHEMA_VERSION = 1
SAMPLE_COUNT = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def line_sha256(line: bytes) -> str:
    return hashlib.sha256(line).hexdigest()


def _video_entry(video_id: int) -> Dict[str, Any]:
    return {
        "video_id": int(video_id),
        "lines": 0,
        "main_decisions": 0,
        "legal_action_count": 0,
        "question_counts": {},
        "min_frame": None,
        "max_frame": None,
        "workload_score": 0,
        "partition_line_sha256_samples": [],
    }


def _sample_update(
    samples: list[dict[str, Any]],
    *,
    rng: random.Random,
    seen: int,
    local_line: int,
    source_line: int,
    offset: int,
    raw: bytes,
) -> None:
    item = {
        "local_line": int(local_line),
        "source_line": int(source_line),
        "offset": int(offset),
        "length": int(len(raw)),
        "sha256": line_sha256(raw),
    }
    if len(samples) < SAMPLE_COUNT:
        samples.append(item)
        return
    replacement = rng.randrange(int(seen) + 1)
    if replacement < SAMPLE_COUNT:
        samples[replacement] = item


def _verify_samples(partition: Path, samples: list[dict[str, Any]]) -> None:
    with partition.open("rb") as handle:
        for sample in samples:
            handle.seek(int(sample["offset"]))
            raw = handle.read(int(sample["length"]))
            if len(raw) != int(sample["length"]) or line_sha256(raw) != sample["sha256"]:
                raise RuntimeError(f"raw partition sample mismatch: {partition}:{sample}")


def _load_cache_keys(cache_root: Path) -> dict[str, list[list[int]]]:
    index_path = cache_root / "index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    result: dict[str, list[list[int]]] = {}
    seen: set[tuple[int, int, int]] = set()
    with index_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            key = (
                int(payload["video_id"]),
                int(payload["frame"]),
                int(payload["view"]),
            )
            if key in seen:
                raise ValueError(f"duplicate perception cache key at line {line_number}: {key}")
            seen.add(key)
            result.setdefault(str(key[0]), []).append(list(key))
    for values in result.values():
        values.sort(key=lambda key: (int(key[1]), int(key[2])))
    return result


def _partition_trace(trace: Path, trace_dir: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    trace_dir.mkdir(parents=True, exist_ok=True)
    handles: dict[int, tuple[Any, Any]] = {}
    stats: dict[str, Any] = {
        "nonempty_lines": 0,
        "video_count": 0,
        "main_decisions": 0,
        "question_counts": {},
    }
    per_video: dict[str, dict[str, Any]] = {}
    rngs: dict[int, random.Random] = {}
    samples: dict[str, list[dict[str, Any]]] = {}

    try:
        with trace.open("rb") as source:
            for source_line, raw in enumerate(source):
                if not raw.strip():
                    continue
                try:
                    event = json.loads(raw)
                    context = event.get("context", {})
                    video_id = int(context["video_id"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid trace JSON at source line {source_line + 1}") from exc
                if video_id not in handles:
                    target = trace_dir / f"video_{video_id:02d}.jsonl"
                    order_target = trace_dir / f"video_{video_id:02d}.orders.jsonl"
                    raw_handle = target.with_suffix(target.suffix + ".tmp").open("wb")
                    order_handle = order_target.with_suffix(order_target.suffix + ".tmp").open("w", encoding="utf-8")
                    handles[video_id] = (raw_handle, order_handle)
                    per_video[str(video_id)] = _video_entry(video_id)
                    samples[str(video_id)] = []
                    rngs[video_id] = random.Random(20261003 + video_id)
                raw_handle, order_handle = handles[video_id]
                entry = per_video[str(video_id)]
                local_line = int(entry["lines"])
                offset = int(raw_handle.tell())
                raw_handle.write(raw)
                order_handle.write(json.dumps({"source_line": int(source_line)}) + "\n")
                entry["lines"] = local_line + 1
                entry["partition_line_sha256_samples"] = samples[str(video_id)]
                _sample_update(
                    samples[str(video_id)],
                    rng=rngs[video_id],
                    seen=local_line,
                    local_line=local_line,
                    source_line=source_line,
                    offset=offset,
                    raw=raw,
                )
                question = str(event.get("question", ""))
                entry["question_counts"][question] = entry["question_counts"].get(question, 0) + 1
                stats["question_counts"][question] = stats["question_counts"].get(question, 0) + 1
                frame = int(context.get("frame", 0))
                entry["min_frame"] = frame if entry["min_frame"] is None else min(entry["min_frame"], frame)
                entry["max_frame"] = frame if entry["max_frame"] is None else max(entry["max_frame"], frame)
                legal_count = len(event.get("legal_actions", ()))
                if is_main_decision(event):
                    entry["main_decisions"] += 1
                    entry["legal_action_count"] += legal_count
                    # This is only a scheduling proxy.  It never changes the
                    # replay workload; every raw line remains in the partition.
                    entry["workload_score"] += max(1, legal_count)
                    stats["main_decisions"] += 1
                stats["nonempty_lines"] += 1
    finally:
        for raw_handle, order_handle in handles.values():
            raw_handle.close()
            order_handle.close()

    for video_id in sorted(per_video, key=lambda value: int(value)):
        raw_target = trace_dir / f"video_{int(video_id):02d}.jsonl"
        order_target = trace_dir / f"video_{int(video_id):02d}.orders.jsonl"
        raw_tmp = raw_target.with_suffix(raw_target.suffix + ".tmp")
        order_tmp = order_target.with_suffix(order_target.suffix + ".tmp")
        os.replace(raw_tmp, raw_target)
        os.replace(order_tmp, order_target)
        _verify_samples(raw_target, samples[video_id])
        per_video[video_id]["partition_line_sha256_samples"] = list(samples[video_id])
    stats["video_count"] = len(per_video)
    return stats, per_video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trace = args.trace.resolve()
    cache = args.cache.resolve()
    output = args.output.resolve()
    manifest_path = output / "partition_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETE":
            print(json.dumps(existing, indent=2, sort_keys=True))
            return
    if not trace.is_file():
        raise FileNotFoundError(trace)
    output.mkdir(parents=True, exist_ok=True)
    trace_dir = output / "trace_by_video"
    stats, per_video = _partition_trace(trace, trace_dir)
    cache_keys = _load_cache_keys(cache)
    for video_id, entry in per_video.items():
        if video_id not in cache_keys:
            raise RuntimeError(f"video {video_id} has trace lines but no cache keys")
        entry["cache_keys"] = len(cache_keys[video_id])
    (output / "cache_keys_by_video.json").write_text(
        json.dumps(cache_keys, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": "COMPLETE",
        "schema_version": SCHEMA_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "trace": str(trace),
        "trace_sha256": "sha256:" + sha256(trace),
        "cache": str(cache),
        "cache_index_sha256": "sha256:" + sha256(cache / "index.jsonl"),
        "output": str(output),
        "trace_by_video": str(trace_dir),
        "stats": stats,
        "videos": per_video,
        "total_partition_lines": sum(int(item["lines"]) for item in per_video.values()),
        "total_main_decisions": sum(int(item["main_decisions"]) for item in per_video.values()),
        "raw_line_partition": True,
        "source_order_sidecars": True,
        "random_exact_checks": SAMPLE_COUNT,
        "sampling": False,
        "truncation": False,
        "semantic_transform": "none; raw source JSONL lines copied by context.video_id",
    }
    if int(manifest["total_partition_lines"]) != int(stats["nonempty_lines"]):
        raise RuntimeError("partition line count does not equal source nonempty line count")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
