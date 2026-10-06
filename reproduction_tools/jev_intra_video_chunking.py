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
import math
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
    initial_key_index: int = 0,
) -> list[dict[str, Any]]:
    """Create stable non-overlapping key ranges for one video."""

    target = int(target_records)
    if target < 1:
        raise ValueError("target_records must be positive")
    keys = [tuple(int(value) for value in key) for key in ordered_keys]
    first_decision_key = max(0, int(initial_key_index))
    if first_decision_key > len(keys):
        raise ValueError("initial_key_index is outside the ordered key range")
    counts = [
        decision_count_for_key(events_by_key.get(key, ()))
        for key in keys[first_decision_key:]
    ]
    chunks: list[dict[str, Any]] = []
    start = first_decision_key
    record_start = 0
    accumulated = 0
    for local_index, count in enumerate(counts):
        index = first_decision_key + local_index
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
        tail_count = sum(counts[start - first_decision_key :])
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
    if chunks and chunks[0]["key_start"] != first_decision_key:
        raise AssertionError("chunk plan has a decision-prefix gap")
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


def _numeric_max_abs(left: Any, right: Any) -> tuple[bool, float]:
    """Compare JSON values with a numeric tolerance and return max error."""

    if isinstance(left, bool) or isinstance(right, bool):
        return (left == right, 0.0 if left == right else math.inf)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not (math.isfinite(float(left)) and math.isfinite(float(right))):
            return (left == right, 0.0 if left == right else math.inf)
        error = abs(float(left) - float(right))
        return (error <= 1e-6, error)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return (False, math.inf)
        equal = True
        maximum = 0.0
        for key in left:
            item_equal, item_error = _numeric_max_abs(left[key], right[key])
            equal = equal and item_equal
            maximum = max(maximum, item_error)
        return (equal, maximum)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return (False, math.inf)
        equal = True
        maximum = 0.0
        for left_item, right_item in zip(left, right):
            item_equal, item_error = _numeric_max_abs(left_item, right_item)
            equal = equal and item_equal
            maximum = max(maximum, item_error)
        return (equal, maximum)
    return (left == right, 0.0 if left == right else math.inf)


def _utility_values(value: Any, path: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    values: dict[tuple[str, ...], Any] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            next_path = path + (str(key),)
            if str(key) == "utility":
                values[next_path] = item
            else:
                values.update(_utility_values(item, next_path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            values.update(_utility_values(item, path + (str(index),)))
    return values


def _record_field_comparison(
    common: Sequence[Any],
    left: Mapping[Any, Mapping[str, Any]],
    right: Mapping[Any, Mapping[str, Any]],
    field: str,
) -> dict[str, Any]:
    mismatches = []
    non_numeric_mismatches = 0
    maximum = 0.0
    for key in common:
        left_value = left[key].get(field)
        right_value = right[key].get(field)
        equal, error = _numeric_max_abs(left_value, right_value)
        maximum = max(maximum, error)
        if not equal:
            mismatches.append(list(key))
            if not math.isfinite(error):
                non_numeric_mismatches += 1
    return {
        "equal_within_1e-6": not mismatches,
        "compared": len(common),
        "mismatches": len(mismatches),
        "mismatch_key_sample": mismatches[:20],
        "max_abs_error": maximum if math.isfinite(maximum) else None,
        "non_numeric_mismatches": non_numeric_mismatches,
    }


def _utility_field_comparison(
    common: Sequence[Any],
    left: Mapping[Any, Mapping[str, Any]],
    right: Mapping[Any, Mapping[str, Any]],
) -> dict[str, Any]:
    mismatches = []
    non_numeric_mismatches = 0
    maximum = 0.0
    for key in common:
        left_value = _utility_values(left[key])
        right_value = _utility_values(right[key])
        equal, error = _numeric_max_abs(left_value, right_value)
        maximum = max(maximum, error)
        if not equal:
            mismatches.append(list(key))
            if not math.isfinite(error):
                non_numeric_mismatches += 1
    return {
        "equal_within_1e-6": not mismatches,
        "compared": len(common),
        "mismatches": len(mismatches),
        "mismatch_key_sample": mismatches[:20],
        "max_abs_error": maximum if math.isfinite(maximum) else None,
        "non_numeric_mismatches": non_numeric_mismatches,
    }


def compare_records_exact(
    single_records: Sequence[Mapping[str, Any]],
    chunked_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare two builds by semantic key and exact canonical JSON value."""

    left_keys = [semantic_record_key(item) for item in single_records]
    right_keys = [semantic_record_key(item) for item in chunked_records]
    left_duplicates = sorted({key for key in left_keys if left_keys.count(key) > 1})
    right_duplicates = sorted({key for key in right_keys if right_keys.count(key) > 1})
    left = {semantic_record_key(item): item for item in single_records}
    right = {semantic_record_key(item): item for item in chunked_records}
    common = sorted(set(left) & set(right))
    mismatches = [
        key
        for key in common
        if json.dumps(left[key], sort_keys=True, separators=(",", ":"), allow_nan=False)
        != json.dumps(right[key], sort_keys=True, separators=(",", ":"), allow_nan=False)
    ]
    best_action_comparison = _record_field_comparison(common, left, right, "best_actions")
    target_probability_comparison = _record_field_comparison(common, left, right, "target_probs")
    state_feature_comparison = _record_field_comparison(
        common,
        {key: {"feature_vector": value.get("state", {}).get("feature_vector")} for key, value in left.items()},
        {key: {"feature_vector": value.get("state", {}).get("feature_vector")} for key, value in right.items()},
        "feature_vector",
    )
    utility_comparison = _utility_field_comparison(common, left, right)
    semantic_keys_exact = set(left) == set(right)
    duplicate_free = not left_duplicates and not right_duplicates
    canonical_sha_equal = canonical_record_digest(single_records) == canonical_record_digest(chunked_records)
    return {
        "schema_version": CHUNKING_SCHEMA_VERSION,
        "status": "PASS"
        if not mismatches and semantic_keys_exact and duplicate_free
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
        "canonical_sha_identical": canonical_sha_equal,
        "semantic_keys_exact": semantic_keys_exact,
        "duplicate_free": duplicate_free,
        "duplicate_semantic_keys_single": [list(key) for key in left_duplicates[:20]],
        "duplicate_semantic_keys_chunked": [list(key) for key in right_duplicates[:20]],
        "field_comparisons": {
            "best_actions": best_action_comparison,
            "target_probs": target_probability_comparison,
            "utilities": utility_comparison,
            "state_features": state_feature_comparison,
        },
    }
