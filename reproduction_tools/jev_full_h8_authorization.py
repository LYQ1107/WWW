"""Fail-closed authorization for the future 24-video canonical H=8 build."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


EXPECTED_VIDEO7_RECORDS = 3337
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


def read_formal_authorization(report_path: Path) -> Mapping[str, Any]:
    """Return a formal report only when every canonicalization gate passes.

    This is intentionally stricter than checking ``status == PASS``.  The
    full rebuild is forbidden if a verifier is incomplete, has a provenance
    mismatch, or only compared a subset of the 3337 video-7 records.
    """

    path = report_path.resolve()
    if not path.is_file():
        raise RuntimeError(f"formal H=8 chunk authorization report is missing: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"formal H=8 chunk authorization report is invalid: {path}") from exc
    if not isinstance(report, Mapping):
        raise RuntimeError(f"formal H=8 chunk authorization report is not an object: {path}")

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
    return report
