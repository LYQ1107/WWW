"""Audit and merge the restart-safe per-video formal H=8 shards.

The fast H=8 worker deliberately writes one official artifact per video:
``video_XX/records.jsonl`` plus ``video_XX/manifest.json``.  This finalizer
is the boundary between those artifacts and the single shared dataset used by
the three policy methods.  It refuses incomplete shards, temporary files,
provenance mismatches, missing decisions, and duplicate event keys.

It is intentionally separate from ``merge_jev_jsonl.py``, whose input
contract is the older ``records.jsonl.manifest.json`` shard format.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from jev_dataset_contract import validate_record


def _normalise_sha256(value: Any) -> str:
    """Compare record and manifest hashes independent of display prefix."""

    text = str(value)
    return text[len("sha256:") :] if text.startswith("sha256:") else text


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _record_key(record: Mapping[str, Any], video_id: int) -> tuple[Any, ...]:
    state = record["state"]
    context = state["online_context"]
    detection_index = context.get("detection_index")
    return (
        int(video_id),
        str(record["sequence"]),
        int(record["frame"]),
        int(record["view"]),
        int(context["event_order"]),
        str(record["question_type"]),
        None if detection_index is None else int(detection_index),
    )


def _iter_validated_records(
    path: Path,
    *,
    video_id: int,
    expected_manifest: Mapping[str, Any],
    seen_keys: set[tuple[Any, ...]],
) -> Iterable[str]:
    checkpoint = _normalise_sha256(expected_manifest["gmt_checkpoint_sha256"])
    expected_horizon = int(expected_manifest["horizon"])
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                validate_record(record, allow_future_gt=True)
                if int(record["horizon"]) != expected_horizon or expected_horizon != 8:
                    raise ValueError("record horizon is not canonical H=8")
                if _normalise_sha256(record["gmt_checkpoint_sha256"]) != checkpoint:
                    raise ValueError("record/checkpoint provenance mismatch")
                context = record["state"]["online_context"]
                if int(context["video_id"]) != int(video_id):
                    raise ValueError("record/video online_context mismatch")
                key = _record_key(record, video_id)
                if key in seen_keys:
                    raise ValueError(f"duplicate event key: {key!r}")
                seen_keys.add(key)
            except Exception as exc:  # noqa: BLE001 - include shard context
                raise ValueError(f"invalid record at {path}:{line_number}: {exc}") from exc
            yield line


def _load_official_video(
    shard_root: Path,
    video_id: int,
    expected_video: Mapping[str, Any],
) -> tuple[Path, Mapping[str, Any]]:
    video_root = shard_root / f"video_{int(video_id):02d}"
    records = video_root / "records.jsonl"
    manifest_path = video_root / "manifest.json"
    if not records.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"video {video_id} is not an official shard: {records} and {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE":
        raise ValueError(f"video {video_id} manifest is not COMPLETE")
    if int(manifest.get("video_id", -1)) != int(video_id):
        raise ValueError(f"video {video_id} manifest has the wrong video_id")
    if bool(manifest.get("sampling")) or bool(manifest.get("truncation")):
        raise ValueError(f"video {video_id} was sampled or truncated")
    if int(manifest.get("horizon", -1)) != 8:
        raise ValueError(f"video {video_id} is not H=8")
    expected_decisions = int(expected_video["main_decisions"])
    if int(manifest.get("source_main_decisions", -1)) != expected_decisions:
        raise ValueError(
            f"video {video_id} source decision mismatch: "
            f"{manifest.get('source_main_decisions')} != {expected_decisions}"
        )
    if int(manifest.get("records", -1)) != expected_decisions:
        raise ValueError(
            f"video {video_id} record count mismatch: "
            f"{manifest.get('records')} != {expected_decisions}"
        )
    return records, manifest


def finalize(
    *,
    shard_root: Path,
    partition_root: Path,
    output: Path,
    output_manifest: Path | None = None,
) -> Mapping[str, Any]:
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {output}")
    partition_manifest = json.loads(
        (partition_root / "partition_manifest.json").read_text(encoding="utf-8")
    )
    if partition_manifest.get("status") != "COMPLETE":
        raise ValueError("partition manifest is not COMPLETE")
    expected_videos = partition_manifest.get("videos") or {}
    if not expected_videos:
        raise ValueError("partition manifest has no videos")
    expected_video_count = int(partition_manifest.get("stats", {}).get("video_count", len(expected_videos)))
    if len(expected_videos) != expected_video_count:
        raise ValueError(
            f"partition video count mismatch: {len(expected_videos)} != {expected_video_count}"
        )

    official: list[tuple[int, Path, Mapping[str, Any]]] = []
    reference: Mapping[str, Any] | None = None
    consistency_fields = (
        "gmt_checkpoint_sha256",
        "source_commit",
        "cache_index_sha256",
        "source_trace_sha256",
        "annotations_sha256",
        "horizon",
        "association_backend",
        "formal_gmt_association_adapter",
        "counterfactual_engine",
        "state_schema_version",
        "utility_definition",
        "sampling",
        "truncation",
    )
    for video_key in sorted(expected_videos, key=lambda value: int(value)):
        video_id = int(video_key)
        records, manifest = _load_official_video(
            shard_root, video_id, expected_videos[video_key]
        )
        if reference is None:
            reference = manifest
        else:
            mismatches = {
                field: (reference.get(field), manifest.get(field))
                for field in consistency_fields
                if reference.get(field) != manifest.get(field)
            }
            if mismatches:
                raise ValueError(
                    f"video {video_id} provenance mismatch: "
                    f"{json.dumps(mismatches, sort_keys=True)}"
                )
        official.append((video_id, records, manifest))

    assert reference is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    seen_keys: set[tuple[Any, ...]] = set()
    record_count = 0
    records_by_video: dict[str, int] = {}
    with temporary.open("w", encoding="utf-8") as target:
        for video_id, records, manifest in official:
            count = 0
            for line in _iter_validated_records(
                records,
                video_id=video_id,
                expected_manifest=manifest,
                seen_keys=seen_keys,
            ):
                target.write(line)
                count += 1
                record_count += 1
            if count != int(manifest["records"]):
                raise ValueError(
                    f"video {video_id} line count mismatch: {count} != {manifest['records']}"
                )
            records_by_video[str(video_id)] = count
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, output)

    expected_total = int(partition_manifest["total_main_decisions"])
    if record_count != expected_total:
        raise ValueError(
            f"full H=8 record count mismatch: {record_count} != {expected_total}"
        )
    report = {
        "status": "PASS",
        "schema_version": 1,
        "output": str(output.resolve()),
        "output_sha256": sha256(output),
        "partition_root": str(partition_root.resolve()),
        "partition_manifest_sha256": sha256(partition_root / "partition_manifest.json"),
        "shard_root": str(shard_root.resolve()),
        "video_count": len(official),
        "video_ids": [video_id for video_id, _, _ in official],
        "record_count": record_count,
        "expected_record_count": expected_total,
        "records_by_video": records_by_video,
        "gmt_checkpoint_sha256": reference["gmt_checkpoint_sha256"],
        "source_commit": reference.get("source_commit"),
        "cache_index_sha256": reference["cache_index_sha256"],
        "source_trace_sha256": reference["source_trace_sha256"],
        "annotations_sha256": reference["annotations_sha256"],
        "horizon": 8,
        "association_backend": reference["association_backend"],
        "counterfactual_engine": reference["counterfactual_engine"],
        "state_schema_version": reference["state_schema_version"],
        "utility_definition": reference["utility_definition"],
        "sampling": False,
        "truncation": False,
        "duplicate_event_keys": 0,
        "source_manifests": [
            {
                "video_id": video_id,
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": sha256(manifest_path),
                "records": str(records.resolve()),
                "records_sha256": sha256(records),
            }
            for video_id, records, manifest in official
            for manifest_path in [shard_root / f"video_{video_id:02d}" / "manifest.json"]
        ],
    }
    if output_manifest is None:
        output_manifest = output.with_suffix(output.suffix + ".manifest.json")
    if output_manifest.exists():
        raise RuntimeError(f"refusing to overwrite output manifest: {output_manifest}")
    atomic_json(output_manifest, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--partition-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path)
    args = parser.parse_args()
    report = finalize(
        shard_root=args.shard_root.resolve(),
        partition_root=args.partition_root.resolve(),
        output=args.output.resolve(),
        output_manifest=args.output_manifest.resolve() if args.output_manifest else None,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
