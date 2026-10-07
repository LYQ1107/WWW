"""Audit whether an older v2 record artifact can be reused after a source fix.

This is an evidence-only bounded comparison.  It deliberately does not alter
the running video01 builder and it never upgrades a sample comparison into a
formal acceptance decision.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
            if limit is not None and len(records) >= int(limit):
                break
    return records


def record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    context = record.get("state", {}).get("online_context", {})
    return (
        record.get("sequence"),
        record.get("frame"),
        record.get("view"),
        record.get("question_type"),
        tuple(record.get("legal_actions", [])),
        context.get("event_order"),
        context.get("detection_index"),
    )


def state_projection(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "state_digest": record.get("state_digest"),
        "state": record.get("state"),
    }


def label_projection(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_outcomes": record.get("action_outcomes"),
        "horizon_outcomes": record.get("horizon_outcomes"),
        "best_actions": record.get("best_actions"),
        "target_probs": record.get("target_probs"),
        "sample_weight": record.get("sample_weight"),
    }


def compact_outcomes(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for action, outcome in (record.get("action_outcomes") or {}).items():
        result[str(action)] = {
            key: outcome.get(key)
            for key in (
                "utility",
                "future_fragmentation",
                "future_identity_switches",
                "future_collisions",
                "memory_contamination",
                "future_correct_identity_duration",
                "sample_weight",
            )
        }
    return result


def git_scope(source_root: Path, old_commit: str, new_commit: str) -> dict[str, Any]:
    command = [
        "git",
        "diff",
        "--numstat",
        old_commit,
        new_commit,
        "--",
        "reproduction_tools/build_jev_counterfactual_v2.py",
        "reproduction_tools/jev_counterfactual_v2.py",
        "reproduction_tools/jev_gmt_association_adapter.py",
        "configs/VISION_test.yaml",
    ]
    result = subprocess.run(
        command,
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3:
            rows.append({"insertions": parts[0], "deletions": parts[1], "path": parts[2]})
    return {
        "from_commit": old_commit,
        "to_commit": new_commit,
        "files_examined": command[command.index("--") + 1 :],
        "numstat": rows,
        "observed_semantic_change": (
            "_controller_semantics now records model.thred_bank as bank_threshold; "
            "reactivation feature construction uses bank_threshold"
        ),
    }


def manifest_value(manifest: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in manifest:
            return manifest[name]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-records", type=Path, required=True)
    parser.add_argument("--new-records", type=Path, required=True)
    parser.add_argument("--historical-records", type=Path)
    parser.add_argument("--repeat-records", type=Path)
    parser.add_argument("--old-manifest", type=Path, required=True)
    parser.add_argument("--new-report", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--old-commit", required=True)
    parser.add_argument("--new-commit", required=True)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sample_size < 1:
        raise ValueError("--sample-size must be positive")

    old_records = load_jsonl(args.old_records, args.sample_size)
    new_records = load_jsonl(args.new_records, args.sample_size)
    historical_records = (
        load_jsonl(args.historical_records, args.sample_size)
        if args.historical_records is not None
        else None
    )
    old_manifest = json.loads(args.old_manifest.read_text(encoding="utf-8"))
    new_report = json.loads(args.new_report.read_text(encoding="utf-8"))

    key_exact = len(old_records) == len(new_records) and all(
        record_key(old) == record_key(new)
        for old, new in zip(old_records, new_records)
    )
    state_equal = [
        state_projection(old) == state_projection(new)
        for old, new in zip(old_records, new_records)
    ]
    labels_equal = [
        label_projection(old) == label_projection(new)
        for old, new in zip(old_records, new_records)
    ]
    mismatches = []
    for index, (old, new) in enumerate(zip(old_records, new_records)):
        if label_projection(old) == label_projection(new):
            continue
        mismatches.append(
            {
                "sample_index": index,
                "key": list(record_key(new)),
                "state_equal": bool(state_equal[index]),
                "old_best_actions": old.get("best_actions"),
                "new_best_actions": new.get("best_actions"),
                "old_target_probs": old.get("target_probs"),
                "new_target_probs": new.get("target_probs"),
                "old_outcomes": compact_outcomes(old),
                "new_outcomes": compact_outcomes(new),
            }
        )

    repeat_equal = None
    repeat_sha = None
    if args.repeat_records is not None:
        repeat = load_jsonl(args.repeat_records, args.sample_size)
        repeat_equal = len(new_records) == len(repeat) and new_records == repeat
        repeat_sha = sha256(args.repeat_records)

    historical_replay = None
    if historical_records is not None:
        historical_replay = {
            "records": str(args.historical_records),
            "records_sha256": sha256(args.historical_records),
            "sample_size": len(historical_records),
            "historical_equals_new_sample": (
                len(historical_records) == len(new_records)
                and historical_records == new_records
            ),
            "historical_equals_old_artifact_sample": (
                len(historical_records) == len(old_records)
                and historical_records == old_records
            ),
            "interpretation": (
                "The committed 60675 historical replay reproduced the current "
                "diagnostic sample, not the old full artifact sample.  Therefore "
                "the old artifact cannot be explained by the tracked 60675->4108 "
                "bank-threshold diff alone; its hidden generation provenance is "
                "not recoverable from the manifest."
            ),
        }

    annotation_path = Path(str(new_report.get("annotations", "")))
    annotation_sha = sha256(annotation_path) if annotation_path.is_file() else None
    old_trace_sha = manifest_value(old_manifest, "trace_sha256")
    new_trace_sha = new_report.get("trace_sha256")
    old_config_sha = manifest_value(old_manifest, "config_sha256")
    new_config_sha = new_report.get("config_sha256")
    old_checkpoint_sha = manifest_value(old_manifest, "gmt_checkpoint_sha256")
    new_checkpoint_sha = new_report.get("checkpoint_sha256")
    old_annotation_sha = manifest_value(old_manifest, "annotations_sha256")

    report = {
        "schema_version": "jev_recompute_scope_audit_v1",
        "status": "DIAGNOSTIC_ONLY",
        "created_utc": "2026-10-07",
        "not_formal_v2_acceptance": True,
        "sample": {
            "old_records": str(args.old_records),
            "new_records": str(args.new_records),
            "sample_size_old": len(old_records),
            "sample_size_new": len(new_records),
            "key_order_exact": key_exact,
            "state_projection_exact_count": sum(state_equal),
            "state_projection_exact": bool(state_equal) and all(state_equal),
            "label_projection_exact_count": sum(labels_equal),
            "label_projection_exact": bool(labels_equal) and all(labels_equal),
            "label_mismatch_count": len(mismatches),
            "first_label_mismatches": mismatches[:5],
            "repeat_new_sample_exact": repeat_equal,
            "repeat_records_sha256": repeat_sha,
            "historical_replay": historical_replay,
        },
        "artifact_sha256": {
            "old_records": sha256(args.old_records),
            "new_records": sha256(args.new_records),
            "old_manifest": sha256(args.old_manifest),
            "new_diagnostic_report": sha256(args.new_report),
        },
        "provenance_comparison": {
            "old_source_commit": old_manifest.get("source_commit"),
            "new_source_commit": new_report.get("source_commit"),
            "old_trace_sha256": old_trace_sha,
            "new_trace_sha256": new_trace_sha,
            "trace_sha256_equal": old_trace_sha == new_trace_sha,
            "old_config_sha256": old_config_sha,
            "new_config_sha256": new_config_sha,
            "config_sha256_equal": old_config_sha == new_config_sha,
            "old_checkpoint_sha256": old_checkpoint_sha,
            "new_checkpoint_sha256": new_checkpoint_sha,
            "checkpoint_sha256_equal": old_checkpoint_sha == new_checkpoint_sha,
            "old_annotations_sha256": old_annotation_sha,
            "new_annotations_sha256": annotation_sha,
            "annotations_sha256_equal": (
                old_annotation_sha is not None
                and annotation_sha is not None
                and old_annotation_sha == annotation_sha
            ),
            "old_cache_index_sha256": manifest_value(
                old_manifest, "perception_cache_index_sha256", "cache_sha256"
            ),
            "new_cache_index_sha256": new_report.get("cache_index_sha256"),
            "old_component_hashes_present": all(
                manifest_value(old_manifest, name) is not None
                for name in (
                    "trajectory_rng_transformer_sha256",
                    "trajectory_rng_counterfactual_engine_sha256",
                    "trajectory_rng_adapter_sha256",
                )
            ),
        },
        "tracked_source_scope": git_scope(
            args.source_root, args.old_commit, args.new_commit
        ),
        "decision": {
            "state_feature_reuse_on_this_sample": (
                "OBSERVED_ONLY" if all(state_equal) else "BLOCKED"
            ),
            "old_reward_and_label_reuse": "BLOCKED",
            "required_recomputation": (
                "Regenerate all action_outcomes, horizon_outcomes, target_probs, "
                "best_actions and sample weights for the v2 artifact."
            ),
            "reason": (
                "The bounded same-input comparison has exact state projections but "
                "non-identical action outcomes/labels.  The old manifest does carry "
                "the input and RNG/component hashes.  A committed 60675 historical "
                "replay reproduces the current sample rather than the old artifact, "
                "so the old artifact's hidden generation provenance is not recoverable "
                "and partial reward/label reuse is not authorized."
            ),
            "full_v2_acceptance": "PENDING_CURRENT_VIDEO01_BUILDER",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
