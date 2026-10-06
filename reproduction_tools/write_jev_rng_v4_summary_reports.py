"""Write compact, reviewable v4 reports from completed pilot artifacts.

This report writer deliberately keeps the runtime pilot separate from the
canonical H=8 decision.  The pilot was run with a preserved legacy trace and
therefore cannot authorize a final rebuild when the state-feature parity gate
is blocked.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Mapping


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def finite_json(value: Any) -> Any:
    """Replace non-finite JSON extensions with strict-JSON nulls."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(finite_json(dict(value)), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def method_runtime_summary(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "metrics": dict(item.get("metrics", {})),
        "delta_vs_gmt_off": dict(item.get("delta_vs_gmt_off", {})),
        "action_counts": dict(item.get("action_counts", {})),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--runtime-report",
        type=Path,
        default=None,
        help="V3 runtime report; defaults to reports/JEV_RUNTIME_STATE_V3/RUNTIME_STATE_TRACKING.json",
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    runtime_path = args.runtime_report or repo / "reports/JEV_RUNTIME_STATE_V3/RUNTIME_STATE_TRACKING.json"
    runtime = read_json(runtime_path)
    parity = read_json(repo / "reports/JEV_RUNTIME_STATE_V3/STATE_FEATURE_PARITY.json")
    three_way_path = repo / "reports/JEV_RNG_V4/SMALL_H8_THREE_WAY_VIDEO06_VIDEO07.json"
    three_way = finite_json(read_json(three_way_path))
    write_json(three_way_path, three_way)
    audit = read_json(repo / "reports/JEV_RNG_V4/OLD_VS_RNG_FIXED_DATASET_AUDIT_VIDEO06_VIDEO07.json")
    repeatability = read_json(repo / "reports/JEV_RNG_V4/REPEATABILITY_ARTIFACTS_VIDEO07.json")
    chunk_plan_path = repo / "reports/JEV_RNG_V4/INTRA_VIDEO_CHUNK_PLAN_VIDEO07.json"
    chunk_probe_path = repo / "reports/JEV_RNG_V4/INTRA_VIDEO_CHUNK_CONTRACT_EQUIVALENCE_VIDEO07.json"
    chunk_plan = read_json(chunk_plan_path) if chunk_plan_path.is_file() else None
    chunk_probe = read_json(chunk_probe_path) if chunk_probe_path.is_file() else None

    runtime_methods = {
        name: method_runtime_summary(item)
        for name, item in runtime.get("methods", {}).items()
    }
    v4_runtime = {
        "schema_version": "jev_rng_v4_small_h8_runtime_tracking_v1",
        "status": "PILOT_FAIL",
        "classification": "SCREENING_ONLY_NOT_FOR_FINAL_SELECTION_NOT_FOR_PAPER_RESULT",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pilot_verdict": "PILOT_FAIL_NO_GO_FOR_CANONICAL_REBUILD",
        "canonical_full_h8_rebuild_authorized": False,
        "video_id": 1,
        "sequence": "00002garden",
        "held_out_tracking_sequence": True,
        "policy_train_sequences": ["00020court3"],
        "policy_val_sequences": ["00017court1"],
        "train_val_tracking_sequence_disjoint": True,
        "new_h8_records_for_tracking_video": 0,
        "legacy_trace_rng_provenance": "BLOCKED_BY_MISSING_LEGACY_RNG_PROVENANCE",
        "runtime_trace": runtime.get("methods", {}).get("gmt_off", {}).get("evaluation", {}),
        "gmt_off_baseline": dict(runtime.get("gmt_off_baseline", {})),
        "methods": runtime_methods,
        "runtime_feature_parity_gate": {
            **dict(runtime.get("runtime_feature_parity_gate", {})),
            "report": "reports/JEV_RUNTIME_STATE_V3/STATE_FEATURE_PARITY.json",
            "observed_report_status": parity.get("status"),
            "observed_report_pass": bool(parity.get("pass", False)),
        },
        "failure_reasons": [
            "generic_mlp_tracking_collapse",
            "full_jev_association_metrics_below_gmt_off",
            "runtime_parity_blocked",
        ],
        "interpretation": (
            "The threshold controller is positive on this screening sequence, but "
            "the required MLP/JEV condition is not met. The MLP collapses association; "
            "JEV lowers HOTA/AssA/IDF1 despite reducing IDSW. The runtime comparison "
            "used a legacy trace and is not a final closed-loop result."
        ),
    }
    write_json(repo / "reports/JEV_RNG_V4/SMALL_H8_RUNTIME_TRACKING.json", v4_runtime)

    jev = three_way["methods"]["jev"]["validation"]
    mlp = three_way["methods"]["question_conditioned_mlp"]["validation"]
    threshold = three_way["methods"]["question_threshold"]["validation"]
    runtime_jev = runtime_methods.get("jev", {})
    runtime_mlp = runtime_methods.get("question_conditioned_mlp", {})
    final = {
        "schema_version": "jev_rng_v4_final_go_no_go_v1",
        "status": "NO_GO",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "canonical_full_h8_rebuild_authorized": False,
        "canonical_baseline": "FROZEN_GMT_BASELINE_REUSED; no complete VISION_test baseline rerun",
        "decision": "PILOT_FAIL_NO_GO_FOR_CANONICAL_REBUILD",
        "gates": {
            "rng_branch_isolation": {"status": "PASS", "report": "reports/JEV_RNG_V4/RNG_BRANCH_ISOLATION.json"},
            "action_order_invariance": {"status": "PASS", "report": "reports/JEV_RNG_V4/ACTION_ORDER_INVARIANCE.json"},
            "reassociate_proposal_reuse": {"status": "PASS", "report": "reports/JEV_RNG_V4/REASSOCIATE_PROPOSAL_REUSE.json"},
            "synthetic_repeatability": {"status": "PASS", "report": "reports/JEV_RNG_V4/REPEATABILITY.json"},
            "worker_independence": {"status": "PASS", "report": "reports/JEV_RNG_V4/WORKER_INDEPENDENCE.json"},
            "video7_artifact_repeatability": {
                "status": repeatability.get("status"),
                "exact_canonical_record_matches": repeatability.get("comparison", {}).get("exact_canonical_record_matches"),
                "raw_records_sha256_equal": repeatability.get("comparison", {}).get("raw_records_sha256_equal"),
                "report": "reports/JEV_RNG_V4/REPEATABILITY_ARTIFACTS_VIDEO07.json",
            },
            "video6_video7_old_vs_new_audit": {
                "status": audit.get("status"),
                "records_compared": audit.get("records_compared"),
                "best_action_agreement_rate": audit.get("best_action_agreement_rate"),
                "strict_old_trace_parity": False,
                "old_trace_rng_provenance": audit.get("old_trace_rng_provenance"),
                "report": "reports/JEV_RNG_V4/OLD_VS_RNG_FIXED_DATASET_AUDIT_VIDEO06_VIDEO07.json",
            },
            "offline_three_way": {
                "status": three_way.get("status"),
                "jev_beats_both_on_val_utility": three_way.get("jev_beats_both_on_val_utility"),
                "jev_val_utility": jev.get("val_utility"),
                "mlp_val_utility": mlp.get("val_utility"),
                "threshold_val_utility": threshold.get("val_utility"),
                "majority_val_utility": three_way.get("baselines", {}).get("majority_action", {}).get("val_utility"),
                "interpretation": "JEV beats the two learned controls by a tiny margin, but matches the majority-action baseline exactly; this is not a meaningful closed-loop authorization signal.",
                "report": "reports/JEV_RNG_V4/SMALL_H8_THREE_WAY_VIDEO06_VIDEO07.json",
            },
            "runtime_state_feature_parity": {
                "status": "FAIL",
                "pass": False,
                "expected_records": parity.get("expected_record_count"),
                "compared_records": parity.get("compared_records"),
                "off_action_mismatches": parity.get("off_action_mismatches"),
                "report": "reports/JEV_RUNTIME_STATE_V3/STATE_FEATURE_PARITY.json",
            },
            "closed_loop_tracking_pilot": {
                "status": "PILOT_FAIL",
                "mlp_delta_ass_a": runtime_mlp.get("delta_vs_gmt_off", {}).get("ΔAssA"),
                "jev_delta_ass_a": runtime_jev.get("delta_vs_gmt_off", {}).get("ΔAssA"),
                "jev_delta_hota": runtime_jev.get("delta_vs_gmt_off", {}).get("ΔHOTA"),
                "report": "reports/JEV_RNG_V4/SMALL_H8_RUNTIME_TRACKING.json",
            },
            "intra_video_chunking_contract_probe": {
                "status": None if chunk_probe is None else chunk_probe.get("status"),
                "classification": None if chunk_probe is None else chunk_probe.get("classification"),
                "exact_record_matches": None
                if chunk_probe is None
                else chunk_probe.get("comparison", {}).get("exact_record_matches"),
                "planned_chunks": None if chunk_plan is None else chunk_plan.get("chunk_count"),
                "formal_gmt_authorization": False,
                "plan_report": "reports/JEV_RNG_V4/INTRA_VIDEO_CHUNK_PLAN_VIDEO07.json",
                "probe_report": "reports/JEV_RNG_V4/INTRA_VIDEO_CHUNK_CONTRACT_EQUIVALENCE_VIDEO07.json",
            },
        },
        "answers": {
            "data_reasonable": "Videos 6/7 v4 records pass schema/provenance and video7 repeats exactly; old-vs-new is screening-only because legacy RNG provenance is unavailable; video1 is still building.",
            "repeatable": "Yes for the completed video7 artifact: 3337/3337 exact canonical records, identical raw SHA, zero semantic deltas.",
            "models_learn": "Offline optimization runs, but the first round is weak: JEV equals the majority baseline on validation utility and its margin over MLP is about 0.0091; this is not sufficient evidence.",
            "runtime_parity": "No final parity claim: the held-out video1 pilot used a preserved legacy trace, with 0 new v4 records available and 24 OFF action mismatches.",
            "closed_loop_signal": "Threshold is positive on this screening sequence; Generic MLP catastrophically collapses association; Full JEV slightly decreases HOTA/AssA/IDF1 while reducing IDSW.",
            "canonical_rebuild": "NO-GO. Do not launch the 24-video canonical H8 rebuild yet; keep the current video1 small-gate worker running.",
            "next_work": "Investigate label/runtime semantics and MLP/JEV action calibration; finish video1; then run the same exact-equivalence verifier with the formal GMT backend before any canonical rebuild authorization.",
        },
        "current_small_gate": {
            "video1": "RUNNING_ON_GPU6_DO_NOT_STOP_OR_MIGRATE",
            "video6": "COMPLETE",
            "video7": "COMPLETE",
            "repeat7": "COMPLETE_AND_EXACTLY_REPEATABLE",
        },
    }
    write_json(repo / "reports/JEV_RNG_V4/FINAL_GO_NO_GO.json", final)
    print(json.dumps({"status": "PASS", "runtime_report": str(repo / "reports/JEV_RNG_V4/SMALL_H8_RUNTIME_TRACKING.json"), "final_decision": final["decision"]}, indent=2))


if __name__ == "__main__":
    main()
