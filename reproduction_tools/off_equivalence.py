"""Strict-but-stable evidence comparison for the OFF replay gates.

The native and traced runs must agree on the discrete tracking trajectory.  A
JSON serializer or CUDA reduction is allowed to perturb an evidence float by
only a very small, explicitly recorded tolerance.  Raw JSON equality remains
an audit field, but it is not the scientific gate by itself.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_ATOL = 1e-6
DEFAULT_RTOL = 1e-6


def _load_prediction_list(path: Path) -> list[Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"prediction payload is not a list: {path}")
    if not all(isinstance(item, Mapping) for item in payload):
        raise ValueError(f"prediction list contains a non-object: {path}")
    return list(payload)


def _required_discrete(item: Mapping[str, Any]) -> tuple[int, int, int]:
    missing = [name for name in ("image_id", "category_id", "track_id") if name not in item]
    if missing:
        raise ValueError(f"prediction is missing discrete fields: {missing}")
    return int(item["image_id"]), int(item["category_id"]), int(item["track_id"])


def _float_values(item: Mapping[str, Any]) -> tuple[float, ...]:
    bbox = item.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("prediction bbox must contain four values")
    return tuple(float(value) for value in bbox) + (float(item.get("score", 0.0)),)


def _group_sorted(items: Iterable[Mapping[str, Any]]) -> dict[tuple[int, int, int], list[tuple[float, ...]]]:
    groups: dict[tuple[int, int, int], list[tuple[float, ...]]] = {}
    for item in items:
        key = _required_discrete(item)
        groups.setdefault(key, []).append(_float_values(item))
    for values in groups.values():
        values.sort()
    return groups


def compare_predictions(
    native_path: Path,
    traced_path: Path,
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> Mapping[str, Any]:
    """Compare prediction evidence with exact discrete trajectory semantics."""

    native = _load_prediction_list(native_path)
    traced = _load_prediction_list(traced_path)
    native_groups = _group_sorted(native)
    traced_groups = _group_sorted(traced)
    discrete_equal = native_groups.keys() == traced_groups.keys()
    float_mismatches = 0
    max_abs = 0.0
    max_rel = 0.0
    if discrete_equal:
        for key in native_groups:
            left = native_groups[key]
            right = traced_groups[key]
            if len(left) != len(right):
                discrete_equal = False
                continue
            for left_values, right_values in zip(left, right):
                for left_value, right_value in zip(left_values, right_values):
                    difference = abs(left_value - right_value)
                    relative = difference / max(abs(left_value), abs(right_value), 1e-30)
                    max_abs = max(max_abs, difference)
                    max_rel = max(max_rel, relative)
                    if not math.isclose(left_value, right_value, abs_tol=atol, rel_tol=rtol):
                        float_mismatches += 1
    else:
        float_mismatches = abs(len(native) - len(traced))

    native_image_ids = {key[0] for key in native_groups}
    traced_image_ids = {key[0] for key in traced_groups}
    native_categories = {key[1] for key in native_groups}
    traced_categories = {key[1] for key in traced_groups}
    native_track_ids = {key[2] for key in native_groups}
    traced_track_ids = {key[2] for key in traced_groups}
    raw_equal = native == traced
    return {
        "discrete_trajectory_equal": bool(discrete_equal),
        "float_evidence_equal": bool(discrete_equal and float_mismatches == 0),
        "raw_json_equal": bool(raw_equal),
        "native_detection_count": len(native),
        "traced_detection_count": len(traced),
        "image_id_set_equal": native_image_ids == traced_image_ids,
        "category_id_set_equal": native_categories == traced_categories,
        "track_id_set_equal": native_track_ids == traced_track_ids,
        "float_mismatch_count": int(float_mismatches),
        "max_float_abs_delta": float(max_abs),
        "max_float_relative_delta": float(max_rel),
        "native_sha256": None,
        "traced_sha256": None,
        "tolerance": {"atol": float(atol), "rtol": float(rtol)},
    }


def _trace_event_key(event: Mapping[str, Any]) -> tuple[Any, ...]:
    context = event.get("context") or {}
    return (
        int(context.get("video_id", -1)),
        int(context.get("frame", -1)),
        int(context.get("view", -1)),
        int(context.get("event_order", -1)),
        str(event.get("question")),
        int(context.get("detection_index", -1)),
    )


def _trace_discrete(event: Mapping[str, Any]) -> tuple[Any, ...]:
    context = event.get("context") or {}
    return (
        _trace_event_key(event),
        tuple(str(value) for value in event.get("legal_actions") or ()),
        event.get("off_action"),
        event.get("proposed_action"),
        event.get("committed_action"),
        context.get("proposal_track_id"),
        context.get("alternate_track_id"),
        context.get("track_id"),
        tuple(sorted((str(key), value) for key, value in (context.get("tracker_state_before") or {}).items() if key != "memory_lengths")),
    )


def _close_float_sequences(left: Any, right: Any, *, atol: float, rtol: float) -> bool:
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _close_float_sequences(a, b, atol=atol, rtol=rtol) for a, b in zip(left, right)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), abs_tol=atol, rel_tol=rtol)
    return left == right


def compare_trace_events(
    native_path: Path,
    traced_path: Path,
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> Mapping[str, Any]:
    """Stream-compare OFF action/assignment events from two runs."""

    events = 0
    mismatches = 0
    future_gt_violations = 0
    float_mismatches = 0
    with native_path.open(encoding="utf-8") as native_handle, traced_path.open(encoding="utf-8") as traced_handle:
        while True:
            native_line = native_handle.readline()
            traced_line = traced_handle.readline()
            if not native_line and not traced_line:
                break
            if not native_line or not traced_line:
                mismatches += 1
                break
            native_event = json.loads(native_line)
            traced_event = json.loads(traced_line)
            events += 1
            if native_event.get("future_gt_access") is not False or traced_event.get("future_gt_access") is not False:
                future_gt_violations += 1
            if _trace_discrete(native_event) != _trace_discrete(traced_event):
                mismatches += 1
            for field in ("state_feature_vector",):
                if not _close_float_sequences(native_event.get(field), traced_event.get(field), atol=atol, rtol=rtol):
                    float_mismatches += 1
            if not _close_float_sequences(
                native_event.get("probabilities"),
                traced_event.get("probabilities"),
                atol=atol,
                rtol=rtol,
            ):
                float_mismatches += 1
    return {
        "action_sequence_equal": bool(events and mismatches == 0),
        "trace_float_evidence_equal": bool(events and float_mismatches == 0),
        "events": int(events),
        "discrete_mismatches": int(mismatches),
        "float_mismatches": int(float_mismatches),
        "future_gt_violations": int(future_gt_violations),
        "tolerance": {"atol": float(atol), "rtol": float(rtol)},
    }
