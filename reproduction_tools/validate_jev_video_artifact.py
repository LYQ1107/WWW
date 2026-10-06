"""Validate one completed corrected H=8 video artifact before runtime replay.

This is a read-only gate.  It verifies the manifest-bound files, JSONL
integrity, semantic-key uniqueness, exact typed-question counts, finite
canonical state features, and the checkpoint/trace/order/record hashes.  It
does not read official TEST annotations or produce tracking metrics.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any


QUESTION_TYPES = (
    "MATCH_DECISION",
    "MEMORY_DECISION",
    "REACTIVATION_DECISION",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def same_hash(actual: str, expected: Any) -> bool:
    if not isinstance(expected, str):
        return False
    left = str(actual).removeprefix("sha256:")
    right = expected.removeprefix("sha256:")
    return left == right


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def semantic_key(record: dict[str, Any]) -> tuple[Any, ...]:
    context = record["state"]["online_context"]
    return (
        str(record.get("sequence")),
        int(context["video_id"]),
        int(context["frame"]),
        int(context["view"]),
        str(record["question_type"]),
        int(context["detection_index"]),
        int(context["event_order"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records_path = args.records.resolve()
    issues: list[str] = []
    if manifest.get("status") != "COMPLETE":
        issues.append(f"manifest status is {manifest.get('status')!r}")
    if int(manifest.get("records", -1)) != int(args.expected_records):
        issues.append("manifest record count mismatch")
    for field in ("sampling", "truncation", "official_test_read"):
        if manifest.get(field) not in (False, None):
            issues.append(f"manifest {field} is not false")
    if int(manifest.get("horizon", -1)) != int(args.horizon):
        issues.append("manifest horizon mismatch")
    if manifest.get("association_backend") != "formal_gmt_transformer":
        issues.append("manifest association backend mismatch")
    if manifest.get("counterfactual_engine") != "cached_perception_mutable_association_v2":
        issues.append("manifest counterfactual engine mismatch")
    if manifest.get("feature_schema_version") != "jev_runtime_state_v2":
        issues.append("manifest feature schema mismatch")
    if manifest.get("trajectory_rng_policy") != "branch_local_explicit_python_random_v1":
        issues.append("manifest RNG policy mismatch")
    if not records_path.is_file():
        issues.append(f"records file missing: {records_path}")

    hash_checks: dict[str, dict[str, Any]] = {}
    path_hash_fields = (
        ("records_artifact", "records_artifact_sha256"),
        ("trace_partition", "trace_partition_sha256"),
        ("source_order_index", "source_order_index_sha256"),
        ("source_trace", "source_trace_sha256"),
        ("gmt_checkpoint", "gmt_checkpoint_sha256"),
        ("annotations", "annotations_sha256"),
    )
    for path_field, hash_field in path_hash_fields:
        value = manifest.get(path_field)
        path = Path(str(value)).resolve() if value else Path("/") / "__missing__"
        item = {"path": str(path), "expected": manifest.get(hash_field)}
        if not path.is_file():
            item.update({"exists": False, "match": False})
            issues.append(f"missing manifest-bound file: {path_field}")
        else:
            actual = sha256(path)
            item.update({"exists": True, "actual": actual, "match": same_hash(actual, manifest.get(hash_field))})
            if not item["match"]:
                issues.append(f"hash mismatch: {path_field}")
        hash_checks[path_field] = item

    question_counts: Counter[str] = Counter()
    record_count = 0
    duplicate_keys = 0
    invalid_lines: list[dict[str, Any]] = []
    nonfinite_records = 0
    canonical_feature_records = 0
    semantic_keys: set[tuple[Any, ...]] = set()
    if records_path.is_file():
        with records_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("record is not an object")
                    question = str(record.get("question_type"))
                    if question not in QUESTION_TYPES:
                        raise ValueError(f"unknown question type: {question}")
                    state = record["state"]
                    context = state["online_context"]
                    features = state["feature_vector"]
                    key = semantic_key(record)
                    semantic_keys.add(key)
                    if len(features) != 64 or not all(math.isfinite(float(item)) for item in features):
                        nonfinite_records += 1
                    if state.get("feature_source") != "canonical_mutable_off_state_v2":
                        raise ValueError("record is not canonical mutable OFF state")
                    if int(record.get("horizon", -1)) != int(args.horizon):
                        raise ValueError("record horizon mismatch")
                    question_counts[question] += 1
                    record_count += 1
                    canonical_feature_records += 1
                except Exception as exc:  # noqa: BLE001 - preserve all failures
                    invalid_lines.append({"line": line_number, "error": str(exc)})

    # The set above intentionally stores full semantic keys; report the
    # duplicate count without retaining another copy of the whole dataset.
    # Re-read only when needed, which keeps the normal path memory-light.
    if records_path.is_file() and record_count:
        seen: set[tuple[Any, ...]] = set()
        with records_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    key = semantic_key(json.loads(line))
                except Exception:
                    continue
                if key in seen:
                    duplicate_keys += 1
                seen.add(key)

    manifest_questions = {
        question: int(manifest.get("records_by_question", {}).get(question, -1))
        for question in QUESTION_TYPES
    }
    question_parity = {
        question: {
            "manifest": manifest_questions[question],
            "records": int(question_counts.get(question, 0)),
            "exact": manifest_questions[question] == int(question_counts.get(question, 0)),
        }
        for question in QUESTION_TYPES
    }
    question_parity["TOTAL"] = {
        "manifest": int(manifest.get("records", -1)),
        "records": record_count,
        "exact": int(manifest.get("records", -1)) == record_count,
    }
    for item in question_parity.values():
        if not item["exact"]:
            issues.append("question count mismatch")
    if record_count != int(args.expected_records):
        issues.append("JSONL record count mismatch")
    if duplicate_keys:
        issues.append("duplicate semantic keys")
    if invalid_lines:
        issues.append("invalid JSONL/record lines")
    if nonfinite_records:
        issues.append("non-finite feature records")

    report = {
        "schema_version": "jev_video_artifact_validation_v1",
        "status": "PASS" if not issues else "FAIL",
        "classification": "PROVENANCE_AND_SCHEMA_GATE_NOT_TRACKING_RESULT",
        "manifest": str(args.manifest.resolve()),
        "records": str(records_path),
        "expected_records": int(args.expected_records),
        "record_count": record_count,
        "invalid_line_count": len(invalid_lines),
        "invalid_lines_sample": invalid_lines[:20],
        "duplicate_semantic_key_count": duplicate_keys,
        "nonfinite_record_count": nonfinite_records,
        "canonical_feature_records": canonical_feature_records,
        "question_parity": question_parity,
        "hash_checks": hash_checks,
        "manifest_source_commit": manifest.get("source_commit"),
        "manifest_checkpoint": manifest.get("gmt_checkpoint"),
        "horizon": int(args.horizon),
        "issues": sorted(set(issues)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "records": record_count, "issues": len(report["issues"])}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
