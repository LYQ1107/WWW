"""Run the corrected-v4 small-video closed-loop comparison fail-closed.

The wrapper consumes only current corrected artifacts and calibrated
single-seed controllers.  It refuses to start if any provenance, runtime
parity, reactivation, stability, formal chunk-equivalence, or training gate is
missing or failed.  Official TEST annotations are never read by the pilot.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]


def float_matches(value: Any, expected: float, tolerance: float = 1e-12) -> bool:
    try:
        observed = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(observed) and abs(observed - float(expected)) <= tolerance


def read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, Mapping) else None


def gate_snapshot(paths: Mapping[str, Path], tolerance: float) -> tuple[dict[str, Any], list[str]]:
    evidence: dict[str, Any] = {}
    failures: list[str] = []

    def load(name: str) -> Mapping[str, Any] | None:
        path = paths[name]
        value = read_json(path)
        evidence[name] = {
            "path": str(path.resolve()),
            "exists": path.is_file(),
            "status": None if value is None else value.get("status"),
        }
        if value is None:
            failures.append(f"missing_or_invalid_{name}")
        return value

    provenance = load("provenance")
    if provenance is not None and provenance.get("status") != "PASS":
        failures.append("video1_provenance_failed")

    candidate = load("candidate")
    if candidate is not None:
        formal_gate = candidate.get("formal_gate", {})
        required = (
            "candidate_id_set_exact",
            "candidate_order_exact_or_canonicalized",
            "candidate_scores_within_frozen_tolerance",
            "chosen_proposal_ids_exact",
            "legacy_off_action_exact",
            "reactivation_decision_count_exact",
        )
        if candidate.get("status") != "PASS" or any(not bool(formal_gate.get(key)) for key in required):
            failures.append("reactivation_candidate_parity_failed")
        if int(formal_gate.get("missing_reactivation_records", -1)) != 0:
            failures.append("reactivation_records_missing")

    parity = load("feature_parity")
    if parity is not None:
        detail = parity.get("parity", {})
        question = detail.get("question_type_parity", {})
        question_keys = ("MATCH_DECISION", "MEMORY_DECISION", "REACTIVATION_DECISION", "TOTAL")
        if (
            parity.get("status") != "PASS"
            or not bool(detail.get("pass"))
            or not bool(detail.get("question_type_parity_pass"))
            or any(not bool(question.get(key, {}).get("exact")) for key in question_keys)
            or int(detail.get("missing_record_count", -1)) != 0
            or int(detail.get("off_action_mismatches", -1)) != 0
            or not float_matches(detail.get("tolerance"), tolerance)
        ):
            failures.append("video1_runtime_feature_parity_failed")

    stability = load("stability")
    if stability is not None:
        if (
            stability.get("status") != "PASS"
            or not bool(stability.get("all_repetitions_pass"))
            or not float_matches(stability.get("recommended_frozen_tolerance"), tolerance)
        ):
            failures.append("runtime_feature_stability_failed")

    formal = load("formal")
    if formal is not None:
        final_gate = formal.get("final_gate", {})
        required = (
            "raw_canonical_records_exact",
            "best_actions_exact",
            "target_probs_within_1e-6",
            "utilities_within_1e-6",
            "state_features_within_1e-6",
            "ranges_no_gap_overlap_duplicate",
            "rng_provenance_exact",
            "canonical_sha_identical",
        )
        if formal.get("status") != "PASS" or any(not bool(final_gate.get(key)) for key in required):
            failures.append("formal_chunk_equivalence_failed")

    training = load("training")
    if training is not None and training.get("status") != "PASS":
        failures.append("corrected_three_way_training_failed")

    return evidence, sorted(set(failures))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", type=int, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--methods-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--provenance-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--feature-parity-report", type=Path, required=True)
    parser.add_argument("--stability-report", type=Path, required=True)
    parser.add_argument("--formal-report", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tolerance", type=float, default=2e-5)
    args = parser.parse_args()

    gate_paths = {
        "provenance": args.provenance_report,
        "candidate": args.candidate_report,
        "feature_parity": args.feature_parity_report,
        "stability": args.stability_report,
        "formal": args.formal_report,
        "training": args.training_report,
    }
    evidence, failures = gate_snapshot(gate_paths, args.tolerance)
    evidence["trace"] = {
        "path": str(args.trace.resolve()),
        "exists": args.trace.is_file(),
    }
    if not args.trace.is_file():
        failures.append("missing_runtime_trace")
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_path = args.output_root / "corrected_v4_tracking.log"
    report: dict[str, Any] = {
        "schema_version": "corrected_v4_small_video_tracking_v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "video_id": int(args.video_id),
        "classification": "CORRECTED_V4_SMALL_H8_CLOSED_LOOP_DIAGNOSTIC_NOT_OFFICIAL_TEST",
        "records": str(args.records.resolve()),
        "trace": str(args.trace.resolve()),
        "methods_root": str(args.methods_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "device": str(args.device),
        "runtime_feature_tolerance": float(args.tolerance),
        "gate_evidence": evidence,
        "gate_failures": failures,
    }
    if failures:
        report.update({"status": "BLOCKED", "tracking_started": False})
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": report["status"], "failures": failures}, indent=2))
        raise SystemExit(2)

    env = os.environ.copy()
    env.update(
        {
            "JEV_VIDEO_ID": str(int(args.video_id)),
            "JEV_TRACE_PATH": str(args.trace.resolve()),
            "JEV_RECORDS_PATH": str(args.records.resolve()),
            "JEV_PILOT_ROOT": str(args.output_root.resolve()),
            "JEV_OFFLINE_ROOT": str((args.output_root / "offline").resolve()),
            "JEV_CACHE_PATH": "/data1/liuyeqiang/WWW/outputs/research_final_v2/off/perception_cache_train",
            "JEV_PILOT_DEVICE": str(args.device),
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": ":".join(
                (
                    str(ROOT),
                    str(ROOT / "reproduction_tools"),
                    str(ROOT / "third_party" / "CenterNet2"),
                    env.get("PYTHONPATH", ""),
                )
            ),
        }
    )
    command = [
        sys.executable,
        "-u",
        str(ROOT / "reproduction_tools" / "run_early_pilot_tracking.py"),
        "--device",
        str(args.device),
        "--feature-source",
        "runtime",
        "--runtime-feature-tolerance",
        str(args.tolerance),
        "--method-checkpoint-root",
        str(args.methods_root.resolve()),
    ]
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    pilot_report_path = args.output_root / "PILOT_TRACKING_THREE_WAY.json"
    pilot_report = read_json(pilot_report_path)
    report.update(
        {
            "status": "PASS" if result.returncode == 0 and pilot_report is not None else "FAIL",
            "tracking_started": True,
            "tracking_exit_code": int(result.returncode),
            "pilot_report": str(pilot_report_path),
            "pilot_status": None if pilot_report is None else pilot_report.get("status"),
            "pilot_verdict": None if pilot_report is None else pilot_report.get("pilot_verdict"),
            "methods": None if pilot_report is None else pilot_report.get("methods"),
        }
    )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "exit_code": result.returncode, "pilot_status": report["pilot_status"]}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(result.returncode or 1)


if __name__ == "__main__":
    main()
