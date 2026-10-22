"""Deterministic taxonomy for auditing PWQT failures."""

from __future__ import annotations


ERROR_TYPES = (
    "DETECTOR_MISS",
    "GEOMETRY_INVALID",
    "WORLD_ALIGNMENT_FAILURE",
    "QUERY_DUPLICATE_BIRTH",
    "QUERY_ASSIGNMENT_ERROR",
    "VISIBILITY_ERROR",
    "PROJECTION_ERROR",
    "IDENTITY_SWITCH",
    "QUERY_CAPACITY_OVERFLOW",
    "UNKNOWN",
)


def classify(record: dict) -> str:
    diagnostics = record.get("query_runtime", record.get("diagnostics", {})) or {}
    if diagnostics.get("query_capacity_overflow"):
        return "QUERY_CAPACITY_OVERFLOW"
    if record.get("geometry_status") in {"INVALID_WORLD_GEOMETRY", "NO_GEOMETRY"}:
        return "GEOMETRY_INVALID"
    alignment = record.get("alignment", {}) or {}
    if alignment and not alignment.get("success", True):
        return "WORLD_ALIGNMENT_FAILURE"
    if diagnostics.get("duplicate_rejections", 0) and diagnostics.get("unmatched_observations"):
        return "QUERY_DUPLICATE_BIRTH"
    if record.get("query_status") == "SMOKE_UNTRAINED":
        return "UNKNOWN"
    return "UNKNOWN"


def summarize(records: list[dict]) -> dict:
    counts = {name: 0 for name in ERROR_TYPES}
    for record in records:
        counts[classify(record)] += 1
    return {"records": len(records), "counts": counts}

