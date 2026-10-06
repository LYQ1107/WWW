"""Deterministic intra-video chunk planning and state-snapshot utilities.

This module is for the future canonical H=8 rebuild only.  The active small
gate remains one worker per video.  A chunk is a contiguous range of ordered
cache keys; boundaries never split a frame/view decision unit.  Each chunk
starts from a state snapshot produced by one production-order GMT-OFF warm-up.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import torch

from jev_counterfactual_v2 import MutableGMTState


CHUNKING_SCHEMA_VERSION = "jev_deterministic_intra_video_chunk_v1"
DECISION_QUESTIONS = {
    "MATCH_DECISION",
    "MEMORY_DECISION",
    "REACTIVATION_DECISION",
}


def semantic_record_key(record: Mapping[str, Any]):
    context = record["state"]["online_context"]
    return (
        int(context["video_id"]),
        int(context["frame"]),
        int(context["view"]),
        str(record["question_type"]),
        int(context["detection_index"]),
    )


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def decision_count_for_key(events: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        int(
            str(event.get("question")) in DECISION_QUESTIONS
            and event.get("context", {}).get("detection_index") is not None
            and bool(event.get("legal_actions"))
        )
        for event in events
    )


def plan_chunks(
    *,
    video_id: int,
    ordered_keys: Sequence[Sequence[int]],
    events_by_key: Mapping[tuple[int, int, int], Sequence[Mapping[str, Any]]],
    target_records: int,
) -> list[dict[str, Any]]:
    """Create stable non-overlapping key ranges for one video."""

    target = int(target_records)
    if target < 1:
        raise ValueError("target_records must be positive")
    keys = [tuple(int(value) for value in key) for key in ordered_keys]
    counts = [decision_count_for_key(events_by_key.get(key, ())) for key in keys]
    chunks: list[dict[str, Any]] = []
    start = 0
    record_start = 0
    accumulated = 0
    for index, count in enumerate(counts):
        accumulated += int(count)
        # A boundary is legal only after the complete current key.  Do not
        # create a zero-record chunk for a run of cache keys without decisions.
        if accumulated >= target and accumulated > 0:
            end = index + 1
            chunks.append(
                {
                    "schema_version": CHUNKING_SCHEMA_VERSION,
                    "video_id": int(video_id),
                    "chunk_index": len(chunks),
                    "key_start": int(start),
                    "key_end": int(end),
                    "record_start": int(record_start),
                    "record_end": int(record_start + accumulated),
                    "decision_count": int(accumulated),
                    "start_key": list(keys[start]),
                    "end_key_exclusive": list(keys[end]) if end < len(keys) else None,
                }
            )
            start = end
            record_start += accumulated
            accumulated = 0
    if start < len(keys):
        tail_count = sum(counts[start:])
        if tail_count > 0:
            chunks.append(
                {
                    "schema_version": CHUNKING_SCHEMA_VERSION,
                    "video_id": int(video_id),
                    "chunk_index": len(chunks),
                    "key_start": int(start),
                    "key_end": int(len(keys)),
                    "record_start": int(record_start),
                    "record_end": int(record_start + tail_count),
                    "decision_count": int(tail_count),
                    "start_key": list(keys[start]),
                    "end_key_exclusive": None,
                }
            )
    if sum(item["decision_count"] for item in chunks) != sum(counts):
        raise AssertionError("chunk plan does not cover all decision records")
    if chunks and chunks[0]["key_start"] != 0:
        raise AssertionError("chunk plan has a prefix gap")
    for left, right in zip(chunks, chunks[1:]):
        if left["key_end"] != right["key_start"]:
            raise AssertionError("chunk plan contains a key-range gap or overlap")
        if left["record_end"] != right["record_start"]:
            raise AssertionError("chunk plan contains a decision-range gap or overlap")
    return chunks


def save_state_snapshot(path: Path, state: MutableGMTState, metadata: Mapping[str, Any]) -> str:
    """Persist an exact CPU clone, including history and trajectory RNG state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHUNKING_SCHEMA_VERSION,
        "metadata": dict(metadata),
        "state": state.clone(),
    }
    temporary = path.with_name(path.name + f".tmp.{path.stat().st_mtime_ns if path.exists() else 'new'}")
    torch.save(payload, temporary)
    temporary.replace(path)
    return file_sha256(path)


def load_state_snapshot(path: Path) -> tuple[MutableGMTState, dict[str, Any]]:
    """Load and validate a chunk start snapshot."""

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("schema_version") != CHUNKING_SCHEMA_VERSION:
        raise ValueError(f"invalid chunk state snapshot: {path}")
    state = payload.get("state")
    metadata = payload.get("metadata")
    if not isinstance(state, MutableGMTState) or not isinstance(metadata, Mapping):
        raise ValueError(f"malformed chunk state snapshot: {path}")
    if state.trajectory_rng_state is None or state.trajectory_rng_seed is None:
        raise ValueError(f"chunk snapshot is missing trajectory RNG provenance: {path}")
    return state, dict(metadata)


def canonical_record_digest(records: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=semantic_record_key):
        payload = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


def compare_records_exact(
    single_records: Sequence[Mapping[str, Any]],
    chunked_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare two builds by semantic key and exact canonical JSON value."""

    left = {semantic_record_key(item): item for item in single_records}
    right = {semantic_record_key(item): item for item in chunked_records}
    if len(left) != len(single_records) or len(right) != len(chunked_records):
        raise ValueError("duplicate semantic record key in comparison input")
    common = sorted(set(left) & set(right))
    mismatches = [
        key
        for key in common
        if json.dumps(left[key], sort_keys=True, separators=(",", ":"), allow_nan=False)
        != json.dumps(right[key], sort_keys=True, separators=(",", ":"), allow_nan=False)
    ]
    return {
        "schema_version": CHUNKING_SCHEMA_VERSION,
        "status": "PASS"
        if not mismatches and set(left) == set(right)
        else "FAIL",
        "single_records": len(single_records),
        "chunked_records": len(chunked_records),
        "common_keys": len(common),
        "only_single": len(set(left) - set(right)),
        "only_chunked": len(set(right) - set(left)),
        "exact_record_matches": len(common) - len(mismatches),
        "record_mismatches": len(mismatches),
        "mismatch_key_sample": [list(key) for key in mismatches[:20]],
        "single_canonical_sha256": canonical_record_digest(single_records),
        "chunked_canonical_sha256": canonical_record_digest(chunked_records),
    }
