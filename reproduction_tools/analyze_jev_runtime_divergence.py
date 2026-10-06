"""Audit why a historical GMT trace cannot be replayed bitwise.

This report is deliberately conservative.  It does not turn a trace-debug or
short smoke replay into a closed-loop tracking result.  The released GMT
transformer uses the process-global Python RNG when assigning trajectory
embedding slots, so an old trace must carry RNG provenance (or the exact slot
assignment) before its association scores can be reproduced in a new
process.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE = Path("/home/liuyeqiang/WWW_jev_full_h8_runtime/partition/trace_by_video/video_08.jsonl")
DEFAULT_RECORDS = Path("/home/liuyeqiang/WWW_jev_full_h8_runtime/pilot/PILOT_TRACKING_TEST.jsonl")
DEFAULT_SMOKE = Path("/home/liuyeqiang/WWW_jev_full_h8_runtime/pilot/PILOT_TRACKING_RAW_SMOKE.json")


def read_trace_probe(path: Path, limit: int = 128) -> dict[str, Any]:
    count = 0
    rng_fields = set()
    candidate_score_events = 0
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        event = json.loads(line)
        count += 1
        rng_fields.update(key for key in event if "rng" in key.lower() or "random" in key.lower())
        context = event.get("context", {})
        rng_fields.update(
            key for key in context if "rng" in key.lower() or "random" in key.lower()
        )
        if isinstance(context.get("candidate_scores"), list):
            candidate_score_events += 1
        if count >= limit:
            break
    return {
        "sampled_events": count,
        "sampled_events_with_candidate_scores": candidate_score_events,
        "rng_or_random_fields": sorted(rng_fields),
        "rng_provenance_present": bool(rng_fields),
    }


def read_smoke(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    method = payload.get("methods", {}).get("gmt_off", {})
    counts = method.get("counts", {})
    return {
        "created_utc": payload.get("created_utc"),
        "max_frame": payload.get("max_frame"),
        "seed": 20261003,
        "payloads": method.get("payloads"),
        "trace_action_records": counts.get("trace_action_records"),
        "off_action_mismatches": counts.get("off_action_mismatches"),
        "feature_parity_records": counts.get("feature_parity_records"),
        "runtime_feature_records": counts.get("runtime_feature_records"),
        "feature_parity_max_abs_error": counts.get("feature_parity_max_abs_error"),
    }


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--smoke", type=Path, default=DEFAULT_SMOKE)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/JEV_RUNTIME_STATE_V3/DIVERGENCE_ANALYSIS.json",
    )
    args = parser.parse_args()

    transformer = ROOT / "gtr/modeling/roi_heads/transformer.py"
    transformer_text = transformer.read_text(encoding="utf-8")
    trace_probe = read_trace_probe(args.trace)
    record_count = sum(1 for line in args.records.open(encoding="utf-8") if line.strip())
    smoke = read_smoke(args.smoke)
    report = {
        "status": "FAIL",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "schema_version": "jev_runtime_state_contract_v3_divergence_v1",
        "classification": "SCREENING_DIAGNOSTIC_NOT_A_PAPER_RESULT",
        "source_revision": git_revision(),
        "trace": str(args.trace),
        "records": str(args.records),
        "record_count": record_count,
        "root_cause": {
            "component": "gtr/modeling/roi_heads/transformer.py",
            "mechanism": "_trajectory_slot_mapping uses process-global random.sample/random.choices on every transformer call",
            "source_uses_process_global_random": (
                "random.sample" in transformer_text and "random.choices" in transformer_text
            ),
            "historical_trace_contains_rng_provenance": trace_probe["rng_provenance_present"],
            "historical_trace_contains_candidate_scores": (
                trace_probe["sampled_events_with_candidate_scores"] > 0
            ),
            "consequence": (
                "A new replay process can have the same mutable tracker state, history order, "
                "checkpoint and perception payloads but still select different learned trajectory "
                "embedding slots, changing association scores and raw score features."
            ),
        },
        "historical_trace_probe": trace_probe,
        "fixed_seed_runtime_probe": smoke,
        "parity_gate": {
            "required_max_abs_error": 1e-6,
            "observed_max_abs_error": (
                smoke.get("feature_parity_max_abs_error") if smoke else None
            ),
            "observed_off_action_mismatches": (
                smoke.get("off_action_mismatches") if smoke else None
            ),
            "pass": False,
            "interpretation": (
                "The short fixed-seed probe is useful for isolating the provenance issue; "
                "it is not a full-sequence runtime result."
            ),
        },
        "affected_feature_names": [
            "accept_score",
            "reassociate_score",
            "raw_traj_score",
            "mean_traj_score",
            "log1p_traj_score",
            "score_minus_threshold",
            "score_over_threshold",
            "raw_score_variance",
        ],
        "scientific_decision": {
            "runtime_controller_claim_allowed": False,
            "learned_controller_tracking_metrics_accepted": False,
            "full_h8_builders_to_stop": False,
            "reason": (
                "The existing Full H=8 builders are independent frozen-data production. "
                "This branch must repair trace provenance and rebuild a matching pilot "
                "before claiming closed-loop controller behavior."
            ),
        },
        "required_remediation": [
            "Make trajectory-slot assignment reproducible and record its seed/state or exact per-window mapping in the formal trace.",
            "Generate a new pilot trace with that provenance using the frozen detector/ReID payloads and checkpoint.",
            "Rebuild pilot counterfactual records from the same trace and rerun the <=1e-6 OFF feature gate.",
            "Only after the gate passes, run Threshold, MLP and JEV true mutated-state tracking.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
