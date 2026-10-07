"""Fail-closed authorization for the future 24-video canonical H=8 build.

The formal chunk-equivalence report is necessary but not sufficient.  A full
H=8 build is allowed only after the corrected held-out video01 pipeline has
passed every runtime gate and the corrected three-way closed-loop comparison
has completed.  This module deliberately treats a missing or stale hard-gate
report as a refusal to authorize.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


EXPECTED_VIDEO7_RECORDS = 3337
VIDEO01_HARD_GATE_FIELDS = (
    "VIDEO01_COMPLETE",
    "VIDEO01_PROVENANCE",
    "VIDEO01_RUNTIME_FEATURE_PARITY",
    "MATCH_COVERAGE",
    "MEMORY_COVERAGE",
    "REACTIVATION_COVERAGE",
    "REACTIVATION_CANDIDATE_PARITY",
    "VIDEO01_NUMERICAL_STABILITY",
    "CORRECTED_THREE_WAY_CLOSED_LOOP_COMPLETE",
)
REQUIRED_FINAL_GATE_FIELDS = (
    "raw_canonical_records_exact",
    "best_actions_exact",
    "target_probs_within_1e-6",
    "utilities_within_1e-6",
    "state_features_within_1e-6",
    "ranges_no_gap_overlap_duplicate",
    "rng_provenance_exact",
    "canonical_sha_identical",
)


def _read_json_object(path: Path, description: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{description} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"{description} is invalid: {path}") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{description} is not an object: {path}")
    return value


def read_video01_hard_gates(report_path: Path | None = None) -> Mapping[str, Any]:
    """Require every corrected-video01 hard gate to be explicitly ``PASS``."""

    if report_path is None:
        report_path = (
            Path(__file__).resolve().parents[1]
            / "reports/JEV_RNG_V4/VIDEO01_CORRECTED_HARD_GATES.json"
        )
    report = _read_json_object(report_path.resolve(), "corrected video01 hard-gate report")
    failures: list[str] = []
    for field in VIDEO01_HARD_GATE_FIELDS:
        if report.get(field) != "PASS":
            failures.append(f"{field}={report.get(field)!r}")
    if report.get("FULL_H8_AUTHORIZED") is not True:
        failures.append(f"FULL_H8_AUTHORIZED={report.get('FULL_H8_AUTHORIZED')!r}")
    if failures:
        raise RuntimeError(
            "corrected video01 hard gates are NOT PASS; " + "; ".join(failures)
        )
    return report


def read_formal_authorization(
    report_path: Path,
    video01_gate_report: Path | None = None,
) -> Mapping[str, Any]:
    """Return a formal report only when every canonicalization gate passes.

    This is intentionally stricter than checking ``status == PASS``.  The
    full rebuild is forbidden if a verifier is incomplete, has a provenance
    mismatch, or only compared a subset of the 3337 video-7 records.
    """

    path = report_path.resolve()
    report = _read_json_object(path, "formal H=8 chunk authorization report")

    failures: list[str] = []
    if report.get("status") != "PASS":
        failures.append(f"status={report.get('status')!r}")
    for field in ("single_records", "chunked_records", "common_keys"):
        if int(report.get(field, -1)) != EXPECTED_VIDEO7_RECORDS:
            failures.append(f"{field}={report.get(field)!r}")
    for field in ("only_single", "only_chunked", "record_mismatches"):
        if int(report.get(field, -1)) != 0:
            failures.append(f"{field}={report.get(field)!r}")
    for field in ("semantic_keys_exact", "duplicate_free", "canonical_sha_identical"):
        if report.get(field) is not True:
            failures.append(f"{field}={report.get(field)!r}")

    range_gate = report.get("chunk_range_gate")
    if not isinstance(range_gate, Mapping) or range_gate.get("pass") is not True:
        failures.append("chunk_range_gate_not_pass")
    rng_gate = report.get("rng_provenance_gate")
    if not isinstance(rng_gate, Mapping) or rng_gate.get("pass") is not True:
        failures.append("rng_provenance_gate_not_pass")
    final_gate = report.get("final_gate")
    if not isinstance(final_gate, Mapping):
        failures.append("final_gate_missing")
    else:
        for field in REQUIRED_FINAL_GATE_FIELDS:
            if final_gate.get(field) is not True:
                failures.append(f"final_gate.{field}={final_gate.get(field)!r}")

    if failures:
        raise RuntimeError(
            "formal H=8 chunk authorization is NOT PASS; "
            + "; ".join(failures)
        )
    # Keep this check after the formal chunk checks so callers get a precise
    # reason for a malformed chunk report before the independent video01 gate.
    read_video01_hard_gates(video01_gate_report)
    return report
