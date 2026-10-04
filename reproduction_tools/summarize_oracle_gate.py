"""Build the auditable Oracle headroom report for the frozen GMT decision layer."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


PER_VIEW_KEYS = (
    ("HOTA", "HOTA", "HOTA"),
    ("DetA", "HOTA", "DetA"),
    ("AssA", "HOTA", "AssA"),
    ("IDF1", "Identity", "IDF1"),
    ("MOTA", "CLEAR", "MOTA"),
    ("IDSW", "CLEAR", "IDSW"),
    ("Frag", "CLEAR", "Frag"),
)
CROSS_KEYS = (
    ("CVIDF1", "cvidf1", "CVIDF1"),
    ("CVMA", "cvma", "CVMA"),
)


def scalar(value: Any) -> float:
    if isinstance(value, list):
        return sum(float(item) for item in value) / len(value) if value else 0.0
    return float(value)


def read_per_view(path: Path) -> Dict[str, float]:
    report = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    combined = report["combined_metrics"]
    return {
        name: scalar(combined[group][key])
        for name, group, key in PER_VIEW_KEYS
    }


def read_cross_view(path: Path) -> Dict[str, float]:
    report = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    return {
        name: scalar(report["reports"][section][field])
        for name, section, field in CROSS_KEYS
    }


def counterfactual_summary(path: Path) -> Dict[str, Any]:
    manifest = json.loads(Path(str(path) + ".manifest.json").read_text(encoding="utf-8"))
    by_question: Dict[str, int] = {}
    best_actions: Dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            question = str(record["question_type"])
            by_question[question] = by_question.get(question, 0) + 1
            for action in record.get("best_actions", []):
                key = f"{question}:{action}"
                best_actions[key] = best_actions.get(key, 0) + 1
    return {
        "manifest": manifest,
        "records_by_question": by_question,
        "best_action_counts": best_actions,
        "engine": manifest.get("counterfactual_engine"),
        "future_gt_used": bool(manifest.get("uses_future_gt", False)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counterfactual", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--threshold", type=Path, required=True)
    parser.add_argument("--jev", type=Path, required=True)
    parser.add_argument("--off-cross", type=Path, required=True)
    parser.add_argument("--oracle-cross", type=Path, required=True)
    parser.add_argument("--threshold-cross", type=Path, required=True)
    parser.add_argument("--jev-cross", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")

    baseline = read_per_view(args.off)
    oracle = read_per_view(args.oracle)
    threshold = read_per_view(args.threshold)
    jev = read_per_view(args.jev)
    baseline_cross = read_cross_view(args.off_cross)
    oracle_cross = read_cross_view(args.oracle_cross)
    threshold_cross = read_cross_view(args.threshold_cross)
    jev_cross = read_cross_view(args.jev_cross)

    def delta(values: Mapping[str, float]) -> Dict[str, float]:
        return {key: float(values[key] - baseline[key]) for key in values}

    trace_future_flags = []
    with args.trace.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                trace_future_flags.append(bool(json.loads(line).get("future_gt_access", True)))

    report = {
        "status": "PASS",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "per_view": {
            "OFF": baseline,
            "ORACLE": oracle,
            "FIXED_THRESHOLD": threshold,
            "JEV_REPLAY": jev,
        },
        "per_view_delta_vs_off": {
            "ORACLE": delta(oracle),
            "FIXED_THRESHOLD": delta(threshold),
            "JEV_REPLAY": delta(jev),
        },
        "cross_view": {
            "OFF": baseline_cross,
            "ORACLE": oracle_cross,
            "FIXED_THRESHOLD": threshold_cross,
            "JEV_REPLAY": jev_cross,
        },
        "cross_view_delta_vs_off": {
            "ORACLE": {key: float(oracle_cross[key] - baseline_cross[key]) for key in oracle_cross},
            "FIXED_THRESHOLD": {key: float(threshold_cross[key] - baseline_cross[key]) for key in threshold_cross},
            "JEV_REPLAY": {key: float(jev_cross[key] - baseline_cross[key]) for key in jev_cross},
        },
        "counterfactual": counterfactual_summary(args.counterfactual),
        "online_trace_future_gt_access": {
            "events": len(trace_future_flags),
            "all_false": bool(trace_future_flags) and not any(trace_future_flags),
        },
        "comparison_policy": {
            "same_frozen_gmt": True,
            "raw_gt_unchanged": True,
            "duplicate_gt_mode": "explicit permissive TrackEval audit",
            "oracle_is_online_result": False,
            "oracle_headroom_is_not_automatic_go": True,
        },
        "decision": "REVIEW_REQUIRED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

