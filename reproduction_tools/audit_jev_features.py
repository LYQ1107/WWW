"""Audit whether online JEV state features are actually populated.

The audit consumes a JSONL trace or counterfactual dataset and reports
non-zero rate, moments, bounded unique counts, exact duplicate streams, and
correlation with a selected legacy score.  It never reads GT to construct a
feature; future-GT fields are treated as metadata and are not inspected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from gtr.modeling.jev_state import feature_names


def _vector(record: Mapping[str, Any]) -> Sequence[float]:
    state = record.get("state")
    if isinstance(state, Mapping) and isinstance(state.get("feature_vector"), list):
        return state["feature_vector"]
    if isinstance(record.get("state_feature_vector"), list):
        return record["state_feature_vector"]
    raise ValueError("record has no state.feature_vector or state_feature_vector")


def _new_stat() -> Dict[str, Any]:
    return {
        "count": 0,
        "nonzero": 0,
        "sum": 0.0,
        "sum_sq": 0.0,
        "min": math.inf,
        "max": -math.inf,
        "unique_values": set(),
        "unique_capped": False,
        "digest": hashlib.sha256(),
        "mean": 0.0,
        "score_mean": 0.0,
        "m2": 0.0,
        "cross": 0.0,
    }


def audit(path: Path, score_feature: str = "accept_score", unique_cap: int = 10000) -> Dict[str, Any]:
    stats: List[Dict[str, Any]] = []
    names: Sequence[str] = ()
    score_index = None
    records = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            values = [float(value) for value in _vector(record)]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"non-finite feature at line {line_number}")
            if not stats:
                names = feature_names(len(values))
                stats = [_new_stat() for _ in values]
                if score_feature not in names:
                    raise ValueError(f"score feature {score_feature!r} is not in {names}")
                score_index = names.index(score_feature)
            if len(values) != len(stats):
                raise ValueError(f"feature dimension changed at line {line_number}")
            score = values[score_index]  # type: ignore[index]
            records += 1
            for index, value in enumerate(values):
                item = stats[index]
                item["count"] += 1
                item["nonzero"] += int(abs(value) > 1e-12)
                item["sum"] += value
                item["sum_sq"] += value * value
                item["min"] = min(item["min"], value)
                item["max"] = max(item["max"], value)
                item["digest"].update(struct.pack("<f", value))
                if not item["unique_capped"]:
                    item["unique_values"].add(round(value, 8))
                    if len(item["unique_values"]) >= unique_cap:
                        item["unique_capped"] = True
                        item["unique_values"].clear()
                n = item["count"]
                delta = value - item["mean"]
                score_delta = score - item["score_mean"]
                item["mean"] += delta / n
                item["score_mean"] += score_delta / n
                item["m2"] += delta * (value - item["mean"])
                item["cross"] += delta * (score - item["score_mean"])

    if not stats:
        raise ValueError("empty feature dataset")
    output_features = []
    for name, item in zip(names, stats):
        variance = item["m2"] / max(1, records)
        score_variance = stats[score_index]["m2"] / max(1, records)  # type: ignore[index]
        correlation = (
            item["cross"] / math.sqrt(max(item["m2"], 0.0) * max(stats[score_index]["m2"], 0.0))  # type: ignore[index]
            if item["m2"] > 0 and score_variance > 0
            else 0.0
        )
        output_features.append(
            {
                "index": len(output_features),
                "feature": name,
                "nonzero_rate": item["nonzero"] / max(1, records),
                "mean": item["sum"] / max(1, records),
                "std": math.sqrt(max(0.0, variance)),
                "min": item["min"],
                "max": item["max"],
                "unique_count": None if item["unique_capped"] else len(item["unique_values"]),
                "unique_count_capped": item["unique_capped"],
                "correlation_with_score": correlation,
                "always_zero": item["nonzero"] == 0,
                "near_zero_variance": variance <= 1e-12,
                "stream_sha256": item["digest"].hexdigest(),
            }
        )
    by_digest: Dict[str, List[str]] = {}
    for item in output_features:
        by_digest.setdefault(item["stream_sha256"], []).append(item["feature"])
    duplicate_groups = [group for group in by_digest.values() if len(group) > 1]
    return {
        "status": "PASS",
        "dataset": str(path.resolve()),
        "records": records,
        "state_dim": len(stats),
        "score_feature": score_feature,
        "feature_schema": list(names),
        "duplicate_stream_groups": duplicate_groups,
        "always_zero_features": [item["feature"] for item in output_features if item["always_zero"]],
        "near_zero_variance_features": [item["feature"] for item in output_features if item["near_zero_variance"]],
        "features": output_features,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-feature", default="accept_score")
    parser.add_argument("--unique-cap", type=int, default=10000)
    args = parser.parse_args()
    if args.unique_cap < 2:
        raise ValueError("unique-cap must be at least two")
    report = audit(args.dataset, args.score_feature, args.unique_cap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
