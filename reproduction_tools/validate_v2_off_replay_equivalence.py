"""Validate the OFF trace contract used by the v2 replay gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--same-gpu-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace = args.trace.resolve()
    same_gpu = json.loads(args.same_gpu_report.resolve().read_text(encoding="utf-8"))
    if same_gpu.get("status") != "PASS" or same_gpu.get("gate_type") != "STRICT_SAME_GPU_OFF_TRAJECTORY":
        raise ValueError("strict same-GPU OFF gate is not PASS")
    counts = {
        "events": 0,
        "invalid_mode": 0,
        "future_gt_access": 0,
        "action_mismatch": 0,
        "probability_mismatch": 0,
        "state_digest_mismatch": 0,
    }
    with trace.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            event = json.loads(line)
            counts["events"] += 1
            if event.get("mode") != "off":
                counts["invalid_mode"] += 1
            if event.get("future_gt_access") is not False:
                counts["future_gt_access"] += 1
            legal = list(event.get("legal_actions") or [])
            off = event.get("off_action")
            if off not in legal or event.get("proposed_action") != off or event.get("committed_action") != off:
                counts["action_mismatch"] += 1
            probabilities = event.get("probabilities") or {}
            if set(probabilities) != set(legal) or any(
                float(probabilities[name]) != float(name == off) for name in legal
            ):
                counts["probability_mismatch"] += 1
            expected = digest(
                {
                    "features": event.get("state_feature_vector"),
                    "question": event.get("question"),
                    "legal_actions": legal,
                    "context": event.get("context") or {},
                }
            )
            if event.get("state_digest") != expected:
                counts["state_digest_mismatch"] += 1
    bad = sum(value for key, value in counts.items() if key != "events" and value)
    report = {
        "status": "PASS" if counts["events"] and not bad else "FAIL",
        "equivalence_mode": "4A_trace_off_contract",
        "gate_type": "TRACE_OFF_CONTRACT_GATE",
        "gate_authority": "FORMAL_REPLAY_PREREQUISITE",
        "official_selection_authority": False,
        "trace": str(trace),
        "trace_sha256": sha256(trace),
        "same_gpu_report": str(args.same_gpu_report.resolve()),
        "same_gpu_prediction_json_equal": same_gpu.get("prediction_json_equal"),
        "same_gpu_discrete_trajectory_equal": (
            same_gpu.get("prediction_comparison", {})
            .get("discrete_trajectory_equal")
        ),
        "counts": counts,
        "future_gt_access": False,
        "line_validation": "streamed_all_nonempty_lines",
    }
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
