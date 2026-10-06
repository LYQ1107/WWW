"""Compare two official per-video H=8 shards produced independently.

The repeatability gate is intentionally stronger than comparing summary counts:
it checks the raw JSONL hash, semantic decision keys, labels, targets, utility
outcomes, and the tracker feature vectors.  Operational manifest fields such as
worker id and output path are reported separately and are not expected to match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


POLICY = "branch_local_explicit_python_random_v1"
MASTER_SEED = 20261006


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def semantic_key(record: Mapping[str, Any], video_id: int) -> tuple[Any, ...]:
    context = record["state"]["online_context"]
    return (
        int(video_id),
        str(record["sequence"]),
        int(record["frame"]),
        int(record["view"]),
        str(record["question_type"]),
        None if context.get("detection_index") is None else int(context["detection_index"]),
        int(context["event_order"]),
    )


def read_records(path: Path, video_id: int) -> tuple[dict[tuple[Any, ...], dict[str, Any]], int]:
    records: dict[tuple[Any, ...], dict[str, Any]] = {}
    invalid = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                key = semantic_key(record, video_id)
                if key in records:
                    raise ValueError(f"duplicate semantic key: {key!r}")
                records[key] = record
            except Exception:  # noqa: BLE001 - report malformed shard lines
                invalid += 1
    return records, invalid


def finite_max_abs(left: Any, right: Any) -> float | None:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return None
    left_value = float(left)
    right_value = float(right)
    if not (math.isfinite(left_value) and math.isfinite(right_value)):
        return None
    return abs(left_value - right_value)


def vector_max_abs(left: Any, right: Any) -> float | None:
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        return None
    values = [finite_max_abs(a, b) for a, b in zip(left, right)]
    values = [value for value in values if value is not None]
    return max(values, default=0.0)


def compare_records(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> tuple[bool, float, float, float]:
    exact = json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )
    target_delta = vector_max_abs(left.get("target_probs"), right.get("target_probs")) or 0.0
    state_delta = vector_max_abs(
        left.get("state", {}).get("feature_vector"),
        right.get("state", {}).get("feature_vector"),
    ) or 0.0
    utility_deltas = []
    for action in set(left.get("action_outcomes", {})) & set(right.get("action_outcomes", {})):
        delta = finite_max_abs(
            left["action_outcomes"][action].get("utility"),
            right["action_outcomes"][action].get("utility"),
        )
        if delta is not None:
            utility_deltas.append(delta)
    return exact, target_delta, state_delta, max(utility_deltas, default=0.0)


def provenance(manifest: Mapping[str, Any], video_id: int) -> dict[str, Any]:
    expected = {
        "status": "COMPLETE",
        "video_id": int(video_id),
        "trajectory_rng_policy": POLICY,
        "trajectory_rng_master_seed": MASTER_SEED,
        "trajectory_rng_video_seed": MASTER_SEED + int(video_id),
        "trajectory_rng_state_cloned_per_counterfactual_branch": True,
        "proposal_reused_across_legal_actions": True,
        "reassociate_reuses_score_matrix": True,
        "second_transformer_call_for_reassociate": False,
    }
    return {
        "pass": all(manifest.get(key) == value for key, value in expected.items()),
        "observed": {key: manifest.get(key) for key in expected},
        "expected": expected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--video-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    left_records_path = args.left / "records.jsonl"
    right_records_path = args.right / "records.jsonl"
    left_manifest_path = args.left / "manifest.json"
    right_manifest_path = args.right / "manifest.json"
    left_records, left_invalid = read_records(left_records_path, args.video_id)
    right_records, right_invalid = read_records(right_records_path, args.video_id)
    common = sorted(set(left_records) & set(right_records), key=str)
    only_left = sorted(set(left_records) - set(right_records), key=str)
    only_right = sorted(set(right_records) - set(left_records), key=str)
    exact_records = 0
    target_deltas: list[float] = []
    state_deltas: list[float] = []
    utility_deltas: list[float] = []
    best_action_mismatches = 0
    question_mismatches: dict[str, int] = {}
    for key in common:
        left = left_records[key]
        right = right_records[key]
        exact, target_delta, state_delta, utility_delta = compare_records(left, right)
        exact_records += int(exact)
        target_deltas.append(target_delta)
        state_deltas.append(state_delta)
        utility_deltas.append(utility_delta)
        if set(left.get("best_actions", ())) != set(right.get("best_actions", ())):
            best_action_mismatches += 1
            question = str(key[4])
            question_mismatches[question] = question_mismatches.get(question, 0) + 1

    left_manifest = json.loads(left_manifest_path.read_text(encoding="utf-8"))
    right_manifest = json.loads(right_manifest_path.read_text(encoding="utf-8"))
    records_hash_equal = sha256(left_records_path) == sha256(right_records_path)
    report = {
        "schema_version": "jev_rng_v4_artifact_repeatability_v1",
        "status": "PASS"
        if records_hash_equal
        and not left_invalid
        and not right_invalid
        and not only_left
        and not only_right
        and exact_records == len(common)
        and best_action_mismatches == 0
        and max(target_deltas, default=0.0) == 0.0
        and max(state_deltas, default=0.0) == 0.0
        and max(utility_deltas, default=0.0) == 0.0
        and provenance(left_manifest, args.video_id)["pass"]
        and provenance(right_manifest, args.video_id)["pass"]
        else "FAIL",
        "video_id": args.video_id,
        "left": {
            "root": str(args.left),
            "records": len(left_records),
            "invalid_lines": left_invalid,
            "records_sha256": sha256(left_records_path),
            "manifest_sha256": sha256(left_manifest_path),
            "manifest_provenance": provenance(left_manifest, args.video_id),
        },
        "right": {
            "root": str(args.right),
            "records": len(right_records),
            "invalid_lines": right_invalid,
            "records_sha256": sha256(right_records_path),
            "manifest_sha256": sha256(right_manifest_path),
            "manifest_provenance": provenance(right_manifest, args.video_id),
        },
        "comparison": {
            "common_semantic_keys": len(common),
            "only_left": len(only_left),
            "only_right": len(only_right),
            "raw_records_sha256_equal": records_hash_equal,
            "exact_canonical_record_matches": exact_records,
            "best_action_mismatches": best_action_mismatches,
            "best_action_mismatch_by_question": question_mismatches,
            "max_target_probability_abs_delta": max(target_deltas, default=0.0),
            "max_state_feature_abs_delta": max(state_deltas, default=0.0),
            "max_action_utility_abs_delta": max(utility_deltas, default=0.0),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **report["comparison"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
