"""Turn the existing video01 candidate-parity evidence into an audit summary."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--native-trace-v3", type=Path, required=True)
    parser.add_argument("--native-trace-v4b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.parity_report.read_text(encoding="utf-8"))
    mismatches = list(source.get("mismatches", []))
    reasons = Counter()
    for item in mismatches:
        values = item.get("reasons")
        if values is None:
            values = [item.get("reason", "unknown")]
        reasons.update(str(value) for value in values)
    score_examples = [
        item
        for item in mismatches
        if "candidate_scores_outside_tolerance" in item.get("reasons", [])
    ][:3]
    missing_examples = [
        item
        for item in mismatches
        if item.get("reason") in {"missing_native_event", "missing_replay_event"}
    ][:12]
    v3_sha = sha256(args.native_trace_v3)
    v4b_sha = sha256(args.native_trace_v4b)
    formal_gate = source.get("formal_gate", {})
    report = {
        "schema_version": "jev_candidate_parity_diagnostic_v1",
        "status": "BLOCKED",
        "not_v2_acceptance": True,
        "source_report": str(args.parity_report),
        "source_report_sha256": sha256(args.parity_report),
        "native_trace_comparison": {
            "v3": str(args.native_trace_v3),
            "v3_sha256": v3_sha,
            "v4b": str(args.native_trace_v4b),
            "v4b_sha256": v4b_sha,
            "byte_identical": v3_sha == v4b_sha,
            "line_count_v3": sum(1 for _ in args.native_trace_v3.open("rb")),
            "line_count_v4b": sum(1 for _ in args.native_trace_v4b.open("rb")),
        },
        "formal_gate": formal_gate,
        "mismatch_accounting": {
            "mismatch_count_capped": source.get("mismatch_count_capped"),
            "missing_native_count": source.get("missing_native_count"),
            "missing_replay_count": source.get("missing_replay_count"),
            "reason_counts_in_reported_mismatches": dict(reasons),
            "score_examples": score_examples,
            "event_key_gap_examples": missing_examples,
        },
        "concrete_findings": {
            "candidate_ids_exact": bool(formal_gate.get("candidate_id_set_exact")),
            "candidate_order_exact_or_canonicalized": bool(
                formal_gate.get("candidate_order_exact_or_canonicalized")
            ),
            "legacy_off_actions_exact": bool(
                formal_gate.get("legacy_off_action_exact")
            ),
            "candidate_scores_pass": bool(
                formal_gate.get("candidate_scores_within_frozen_tolerance")
            ),
            "chosen_proposal_ids_pass": bool(
                formal_gate.get("chosen_proposal_ids_exact")
            ),
            "event_key_coverage_pass": (
                source.get("missing_native_count", 0) == 0
                and source.get("missing_replay_count", 0) == 0
            ),
            "bank_threshold_fix_sufficient": False,
            "classification": (
                "STALE_REPLAY_GATE_BLOCKED: candidate IDs/order and legacy OFF "
                "actions align, but event coverage, candidate scores, and selected "
                "proposal IDs do not.  The old replay is not a valid parity witness."
            ),
            "fresh_v2_candidate_parity": "PENDING_V2_RECORD_ARTIFACT",
        },
        "required_next_check": (
            "After current video01 v2 finishes, run parity against the v2-bound "
            "records and inspect mismatch-covering events before any controller "
            "selection or full H8 authorization."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
