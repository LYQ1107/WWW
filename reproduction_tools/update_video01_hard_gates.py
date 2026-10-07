"""Publish the consolidated corrected-video01 authorization gates.

This report is intentionally fail-closed.  It is the only video01 status
object consumed by :mod:`jev_full_h8_authorization`; a missing input, a failed
input, or an incomplete question-type coverage table cannot become a PASS.
The report is diagnostic metadata, not a tracking result.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Mapping


QUESTION_TYPES = ("MATCH_DECISION", "MEMORY_DECISION", "REACTIVATION_DECISION")


def read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, Mapping) else None


def status_pass(value: Mapping[str, Any] | None) -> bool:
    return value is not None and value.get("status") == "PASS"


def question_table(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return {}
    for candidate in (
        value.get("question_parity"),
        value.get("question_type_parity"),
        (value.get("parity") or {}).get("question_type_parity")
        if isinstance(value.get("parity"), Mapping)
        else None,
    ):
        if isinstance(candidate, Mapping):
            return candidate
    return {}


def question_count(item: Mapping[str, Any]) -> int:
    for key in ("records", "runtime", "expected", "manifest"):
        try:
            return int(item[key])
        except (KeyError, TypeError, ValueError):
            continue
    return 0


def coverage_status(
    provenance: Mapping[str, Any] | None,
    parity: Mapping[str, Any] | None,
    question: str,
) -> tuple[str, dict[str, Any]]:
    prov_item = question_table(provenance).get(question)
    parity_item = question_table(parity).get(question)
    prov_ok = isinstance(prov_item, Mapping) and prov_item.get("exact") is True
    parity_ok = isinstance(parity_item, Mapping) and parity_item.get("exact") is True
    count = max(
        question_count(prov_item) if isinstance(prov_item, Mapping) else 0,
        question_count(parity_item) if isinstance(parity_item, Mapping) else 0,
    )
    passed = status_pass(provenance) and status_pass(parity) and prov_ok and parity_ok and count > 0
    return (
        "PASS" if passed else "PENDING",
        {
            "provenance_exact": prov_ok,
            "runtime_exact": parity_ok,
            "records_observed": count,
            "requires_nonzero_coverage": True,
        },
    )


def candidate_pass(value: Mapping[str, Any] | None) -> bool:
    if not status_pass(value):
        return False
    formal = value.get("formal_gate")
    if not isinstance(formal, Mapping):
        return False
    required = (
        "candidate_id_set_exact",
        "candidate_order_exact_or_canonicalized",
        "candidate_scores_within_frozen_tolerance",
        "chosen_proposal_ids_exact",
        "legacy_off_action_exact",
        "reactivation_decision_count_exact",
    )
    return all(formal.get(key) is True for key in required) and int(
        formal.get("missing_reactivation_records", -1)
    ) == 0


def stability_pass(value: Mapping[str, Any] | None) -> bool:
    return (
        status_pass(value)
        and value.get("all_repetitions_pass") is True
        and value.get("recommended_frozen_tolerance") is not None
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report_root = root / "reports/JEV_RNG_V4"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/liuyeqiang/WWW_jev_rng_v4_runtime/formal_current_head_corrected_full/video01_records.jsonl.manifest.json"),
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=report_root / "CURRENT_HEAD_VIDEO01_PROVENANCE.json",
    )
    parser.add_argument(
        "--parity",
        type=Path,
        default=report_root / "RUNTIME_FEATURE_PARITY_VIDEO01_CURRENT_HEAD.json",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=report_root / "REACTIVATION_CANDIDATE_PARITY_CURRENT_HEAD_VIDEO01.json",
    )
    parser.add_argument(
        "--stability",
        type=Path,
        default=report_root / "RUNTIME_FEATURE_PARITY_STABILITY_VIDEO01_CURRENT_HEAD.json",
    )
    parser.add_argument(
        "--closed-loop",
        type=Path,
        default=report_root / "CURRENT_HEAD_VIDEO01_THREE_WAY_TRACKING.json",
    )
    parser.add_argument("--output", type=Path, default=report_root / "VIDEO01_CORRECTED_HARD_GATES.json")
    args = parser.parse_args()

    manifest = read_json(args.manifest.resolve())
    provenance = read_json(args.provenance.resolve())
    parity = read_json(args.parity.resolve())
    candidate = read_json(args.candidate.resolve())
    stability = read_json(args.stability.resolve())
    closed_loop = read_json(args.closed_loop.resolve())

    video01_complete = bool(
        manifest is not None
        and manifest.get("status") in {"PASS", "COMPLETE"}
        and int(manifest.get("records", -1)) == 8995
    )
    video01_provenance = status_pass(provenance)
    runtime_feature_parity = bool(
        status_pass(parity)
        and isinstance(parity.get("parity"), Mapping)
        and parity["parity"].get("pass") is True
        and parity["parity"].get("question_type_parity_pass") is True
        and int(parity["parity"].get("missing_record_count", -1)) == 0
        and int(parity["parity"].get("off_action_mismatches", -1)) == 0
    )
    coverage: dict[str, dict[str, Any]] = {}
    coverage_statuses: dict[str, str] = {}
    for question, field in (
        ("MATCH_DECISION", "MATCH_COVERAGE"),
        ("MEMORY_DECISION", "MEMORY_COVERAGE"),
        ("REACTIVATION_DECISION", "REACTIVATION_COVERAGE"),
    ):
        value, detail = coverage_status(provenance, parity, question)
        coverage_statuses[field] = value
        coverage[field] = detail

    gates: dict[str, Any] = {
        "VIDEO01_COMPLETE": "PASS" if video01_complete else "PENDING",
        "VIDEO01_PROVENANCE": "PASS" if video01_provenance else "PENDING",
        "VIDEO01_RUNTIME_FEATURE_PARITY": "PASS" if runtime_feature_parity else "PENDING",
        **coverage_statuses,
        "REACTIVATION_CANDIDATE_PARITY": "PASS" if candidate_pass(candidate) else "PENDING",
        "VIDEO01_NUMERICAL_STABILITY": "PASS" if stability_pass(stability) else "PENDING",
        "CORRECTED_THREE_WAY_CLOSED_LOOP_COMPLETE": (
            "PASS"
            if status_pass(closed_loop) and closed_loop.get("tracking_started") is True
            else "PENDING"
        ),
    }
    all_pass = all(value == "PASS" for value in gates.values())
    report = {
        "schema_version": "jev_video01_corrected_hard_gates_v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "PASS" if all_pass else "PENDING",
        "classification": "VIDEO01_AUTHORIZATION_GATE_NOT_TRACKING_RESULT",
        **gates,
        "FULL_H8_AUTHORIZED": bool(all_pass),
        "authorization_decision": "AUTHORIZED" if all_pass else "NOT_AUTHORIZED",
        "question_coverage": coverage,
        "inputs": {
            "manifest": str(args.manifest.resolve()),
            "provenance": str(args.provenance.resolve()),
            "runtime_feature_parity": str(args.parity.resolve()),
            "reactivation_candidate_parity": str(args.candidate.resolve()),
            "numerical_stability": str(args.stability.resolve()),
            "corrected_three_way_closed_loop": str(args.closed_loop.resolve()),
        },
        "policy": {
            "full_h8_requires_every_named_gate": True,
            "missing_or_failed_gate_is_authorization_failure": True,
            "video01_record_count": 8995,
            "required_question_types": list(QUESTION_TYPES),
        },
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "FULL_H8_AUTHORIZED": report["FULL_H8_AUTHORIZED"]}, indent=2))


if __name__ == "__main__":
    main()
