"""Build an isolated early H=8 pilot snapshot and audit it.

This script deliberately treats the formal H=8 builders as read-only inputs.
For running shards it records the byte size first and reads no byte beyond
that boundary.  The source files are never opened for writing and are never
truncated.  The pilot copies only validated JSON records into its own runtime
directory, with video/sequence-disjoint policy and tracking partitions.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path("/home/liuyeqiang/WWW_jev_full_h8_runtime")
PILOT = RUNTIME / "pilot"
QUEUE = RUNTIME / "queue_state.json"
CHECKPOINT = Path("/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth")
CHECKPOINT_SHA256 = "cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8"
HORIZON = 8
SEED = 20261003
UTILITY_DEFINITION = (
    "future_correct_identity_duration - 0.5*future_identity_switches - "
    "0.25*future_fragmentation - 0.5*future_collisions - memory_contamination; "
    "sample_weight=0 for uninformative futures"
)
EXPECTED_BACKEND = "formal_gmt_transformer"
EXPECTED_ENGINE = "cached_perception_mutable_association_v2"
QUESTION_NAMES = (
    "MATCH_DECISION",
    "MEMORY_DECISION",
    "REACTIVATION_DECISION",
)
ACTION_NAMES = (
    "ACCEPT_CURRENT",
    "REASSOCIATE",
    "START_NEW",
    "WRITE_MEMORY",
    "SKIP_MEMORY",
    "REACTIVATE_OLD",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def finite_tree(value: Any) -> bool:
    """Return false for any non-finite numeric value in a JSON tree."""

    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(finite_tree(child) for child in value.values())
    if isinstance(value, Sequence):
        return all(finite_tree(child) for child in value)
    return False


def expected_utility(outcome: Mapping[str, Any]) -> float:
    return (
        float(outcome["future_correct_identity_duration"])
        - 0.5 * float(outcome["future_identity_switches"])
        - 0.25 * float(outcome["future_fragmentation"])
        - 0.5 * float(outcome["future_collisions"])
        - float(outcome["memory_contamination"])
    )


def bounded_lines(path: Path) -> tuple[int, str, list[bytes], int, bool]:
    """Read only the bytes visible at the initial stat boundary.

    Returns ``(size, sha256, complete_lines, dropped_bytes, dropped_line)``.
    A final non-newline-terminated fragment is deliberately excluded from the
    pilot copy.  This function never writes to ``path``.
    """

    size = path.stat().st_size
    with path.open("rb") as handle:
        data = handle.read(size)
    if len(data) != size:
        raise IOError(f"short read for snapshot boundary {path}: {len(data)} != {size}")
    digest = sha256_bytes(data)
    pieces = data.split(b"\n")
    dropped_bytes = 0
    dropped_line = False
    if data.endswith(b"\n"):
        pieces = pieces[:-1]
    elif pieces:
        dropped = pieces.pop()
        dropped_bytes = len(dropped)
        dropped_line = bool(dropped.strip())
    complete = [piece for piece in pieces if piece.strip()]
    return size, digest, complete, dropped_bytes, dropped_line


def current_sources(queue_path: Path) -> list[dict[str, Any]]:
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    sources: list[dict[str, Any]] = []
    for video_key, item in sorted(queue["videos"].items(), key=lambda pair: int(pair[0])):
        video_id = int(video_key)
        status = str(item.get("status"))
        if status == "COMPLETE":
            path = Path(str(item["output_root"])) / "records.jsonl"
            source_kind = "COMPLETE"
            pid = None
        elif status == "RUNNING":
            pid = int(item["pid"])
            path = Path(str(item["output_root"])) / f"records.jsonl.tmp.{pid}"
            source_kind = "RUNNING_PID_SNAPSHOT"
        else:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"queue source is missing: video={video_id} path={path}")
        sources.append(
            {
                "video_id": video_id,
                "status": status,
                "pid": pid,
                "source_kind": source_kind,
                "path": str(path),
                "queue_updated_utc": queue.get("updated_utc"),
            }
        )
    if not sources:
        raise RuntimeError("no COMPLETE/RUNNING formal H=8 shard is available")
    return sources


def import_contract():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "reproduction_tools"))
    sys.path.insert(0, str(ROOT / "third_party" / "CenterNet2"))
    from jev_dataset_contract import validate_record

    return validate_record


def validate_pilot_record(record: Mapping[str, Any], validate_record) -> tuple[bool, list[str]]:
    errors: list[str] = []
    try:
        validate_record(record, allow_future_gt=True)
    except Exception as exc:  # validation evidence is retained in the audit
        errors.append(f"contract:{exc}")
    if int(record.get("horizon", -1)) != HORIZON:
        errors.append(f"horizon:{record.get('horizon')}")
    if record.get("gmt_checkpoint_sha256") != CHECKPOINT_SHA256:
        errors.append("checkpoint_sha256")
    state = record.get("state")
    if not isinstance(state, Mapping):
        errors.append("state:not_mapping")
    else:
        if state.get("association_backend") != EXPECTED_BACKEND:
            errors.append("state.association_backend")
        if state.get("counterfactual_engine") != EXPECTED_ENGINE:
            errors.append("state.counterfactual_engine")
        if not isinstance(state.get("feature_vector"), list) or not state.get("feature_vector"):
            errors.append("state.feature_vector")
        if not finite_tree(state):
            errors.append("state.nonfinite")
        context = state.get("online_context")
        if not isinstance(context, Mapping):
            errors.append("state.online_context")
    if str(record.get("question_type")) not in QUESTION_NAMES:
        errors.append("question_type")
    legal = list(record.get("legal_actions", ()))
    outcomes = record.get("action_outcomes", {})
    if not isinstance(outcomes, Mapping) or set(outcomes) != set(legal):
        errors.append("action_outcomes")
    else:
        for action in legal:
            outcome = outcomes[action]
            if not isinstance(outcome, Mapping) or not finite_tree(outcome):
                errors.append(f"outcome:{action}:nonfinite")
                continue
            try:
                utility = float(outcome["utility"])
                if not math.isfinite(utility):
                    errors.append(f"outcome:{action}:utility_nonfinite")
                elif abs(utility - expected_utility(outcome)) > 1e-5:
                    errors.append(f"outcome:{action}:utility_definition")
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"outcome:{action}:utility:{exc}")
    target = record.get("target_probs", ())
    if not isinstance(target, Sequence) or any(
        not math.isfinite(float(value)) for value in target
    ):
        errors.append("target_probs_nonfinite")
    return not errors, errors


def snapshot_and_audit() -> dict[str, Any]:
    validate_record = import_contract()
    created = utc_now()
    snapshot_root = PILOT / "snapshot"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    sources = current_sources(QUEUE)
    provenance: list[dict[str, Any]] = []
    valid_records: list[dict[str, Any]] = []
    invalid_samples: list[dict[str, Any]] = []
    source_valid_counts: dict[str, int] = {}
    source_invalid_counts: dict[str, int] = {}
    sequence_by_video: dict[str, set[str]] = {}
    state_dims: set[int] = set()

    for source in sources:
        path = Path(source["path"])
        size, source_hash, lines, dropped_bytes, dropped_line = bounded_lines(path)
        copy_path = snapshot_root / f"video_{int(source['video_id']):02d}.jsonl"
        valid_here = 0
        invalid_here = 0
        with copy_path.open("w", encoding="utf-8") as out:
            for line_number, raw in enumerate(lines, 1):
                try:
                    record = json.loads(raw.decode("utf-8"))
                    ok, errors = validate_pilot_record(record, validate_record)
                except Exception as exc:
                    record = None
                    ok = False
                    errors = [f"json:{exc}"]
                if not ok:
                    invalid_here += 1
                    if len(invalid_samples) < 100:
                        invalid_samples.append(
                            {
                                "video_id": int(source["video_id"]),
                                "line": line_number,
                                "errors": errors,
                            }
                        )
                    continue
                record = dict(record)
                out.write(json.dumps(record, sort_keys=True) + "\n")
                valid_here += 1
                valid_records.append(record)
                sequence = str(record["sequence"])
                sequence_by_video.setdefault(str(source["video_id"]), set()).add(sequence)
                state_dims.add(len(record["state"]["feature_vector"]))
        source_valid_counts[str(source["video_id"])] = valid_here
        source_invalid_counts[str(source["video_id"])] = invalid_here
        source.update(
            {
                "snapshot_size_bytes": size,
                "snapshot_source_sha256": source_hash,
                "snapshot_lines_seen": len(lines),
                "snapshot_valid_records": valid_here,
                "snapshot_invalid_records": invalid_here,
                "dropped_incomplete_bytes": dropped_bytes,
                "dropped_incomplete_line": dropped_line,
                "snapshot_copy": str(copy_path),
            }
        )

    if not valid_records:
        raise RuntimeError("pilot snapshot contains no valid records")

    # Video 08 is the held-out pilot tracking sequence.  It is never written
    # to the policy dataset.  Validation is a whole video/sequence group, not
    # a random record split.
    test_video = "8"
    test_sequences = sorted(sequence_by_video.get(test_video, set()))
    if not test_sequences:
        raise RuntimeError("video_08 is not present in the read-only pilot snapshot")
    policy_videos = sorted(key for key in sequence_by_video if key != test_video)
    if len(policy_videos) < 2:
        raise RuntimeError("pilot needs at least two non-test videos for train/val")
    val_video = policy_videos[0]
    val_sequences = sorted(sequence_by_video[val_video])
    train_sequences = sorted(
        sequence
        for video, sequences in sequence_by_video.items()
        if video not in {test_video, val_video}
        for sequence in sequences
    )
    if not train_sequences or not val_sequences:
        raise RuntimeError("empty sequence-disjoint policy train/val split")

    policy_path = PILOT / "PILOT_POLICY.jsonl"
    tracking_path = PILOT / "PILOT_TRACKING_TEST.jsonl"
    train_count = val_count = test_count = 0
    with policy_path.open("w", encoding="utf-8") as policy, tracking_path.open(
        "w", encoding="utf-8"
    ) as tracking:
        for record in valid_records:
            sequence = str(record["sequence"])
            line = json.dumps(record, sort_keys=True) + "\n"
            if sequence in test_sequences:
                tracking.write(line)
                test_count += 1
            else:
                policy.write(line)
                if sequence in val_sequences:
                    val_count += 1
                else:
                    train_count += 1

    split_payload = {
        "protocol": "EARLY_CLOSED_LOOP_PILOT",
        "created_utc": created,
        "seed": SEED,
        "train_sequences": train_sequences,
        "val_sequences": val_sequences,
        "tracking_test_sequences": test_sequences,
        "official_test_used_for_search": False,
        "official_test_access": "BLOCKED_BEFORE_FINAL_SELECTION_LOCK",
        "official_test_files": [],
        "split_rule": "whole VIDEO/SEQUENCE groups; no random record split",
    }
    write_json(PILOT / "PILOT_POLICY_SPLIT.json", split_payload)

    questions = {name: 0 for name in QUESTION_NAMES}
    best_counts = {name: 0 for name in ACTION_NAMES}
    legal_counts = {name: 0 for name in ACTION_NAMES}
    utility_values: list[float] = []
    tie_records = 0
    utility_tie_records = 0
    zero_weight = 0
    finite_state = 0
    sequence_counts: dict[str, int] = {}
    video_counts: dict[str, int] = {}
    for record in valid_records:
        question = str(record["question_type"])
        questions[question] = questions.get(question, 0) + 1
        for action in record["legal_actions"]:
            legal_counts[action] = legal_counts.get(action, 0) + 1
        for action in record["best_actions"]:
            best_counts[action] = best_counts.get(action, 0) + 1
        if len(record["best_actions"]) > 1:
            tie_records += 1
        utilities = [float(record["action_outcomes"][action]["utility"]) for action in record["legal_actions"]]
        utility_values.extend(utilities)
        if max(utilities) - min(utilities) <= 1e-8:
            utility_tie_records += 1
        if float(record.get("sample_weight", 1.0)) <= 0:
            zero_weight += 1
        if finite_tree(record["state"]):
            finite_state += 1
        sequence = str(record["sequence"])
        sequence_counts[sequence] = sequence_counts.get(sequence, 0) + 1
        video_id = str(record["state"]["online_context"].get("video_id"))
        video_counts[video_id] = video_counts.get(video_id, 0) + 1

    total = len(valid_records)
    best_mode_name, best_mode_count = max(best_counts.items(), key=lambda pair: pair[1])
    suspicious_reasons: list[str] = []
    best_mode_share = best_mode_count / max(1, sum(best_counts.values()))
    utility_tie_rate = utility_tie_records / max(1, total)
    if best_mode_share > 0.95:
        suspicious_reasons.append(f"best_action_mode_share={best_mode_share:.6f}>0.95")
    if utility_tie_rate > 0.90:
        suspicious_reasons.append(f"utility_tie_rate={utility_tie_rate:.6f}>0.90")
    if finite_state != total:
        suspicious_reasons.append(f"nonfinite_state_records={total - finite_state}")
    if invalid_samples:
        suspicious_reasons.append(f"invalid_records={sum(source_invalid_counts.values())}")
    status = "PILOT_DATASET_SUSPICIOUS" if suspicious_reasons else "PASS"
    audit = {
        "status": status,
        "pilot_classification": "SCREENING_ONLY_NOT_FOR_FINAL_SELECTION_NOT_FOR_PAPER_RESULT",
        "created_utc": created,
        "snapshot_boundary": "per-source os.stat size before read; running final incomplete line dropped from pilot copy only",
        "formal_builders_modified": False,
        "source_queue": str(QUEUE),
        "source_queue_updated_utc": json.loads(QUEUE.read_text(encoding="utf-8")).get("updated_utc"),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "record_count": total,
        "snapshot_source_count": len(sources),
        "source_provenance": sources,
        "valid_record_count": total,
        "invalid_record_count": sum(source_invalid_counts.values()),
        "invalid_samples": invalid_samples,
        "validation": {
            "json_parse": "PASS" if not invalid_samples else "FAIL_FOR_EXCLUDED_LINES",
            "formal_validate_record_allow_future_gt": "PASS" if not invalid_samples else "FAIL_FOR_EXCLUDED_LINES",
            "horizon_H": HORIZON,
            "checkpoint_sha256": "PASS",
            "association_backend": EXPECTED_BACKEND,
            "counterfactual_engine": EXPECTED_ENGINE,
            "utility_definition": UTILITY_DEFINITION,
            "state_dimensions": sorted(state_dims),
        },
        "question_distribution": questions,
        "legal_action_distribution": legal_counts,
        "best_action_distribution": best_counts,
        "utility_distribution": {
            "count": len(utility_values),
            "min": min(utility_values),
            "max": max(utility_values),
            "mean": statistics.fmean(utility_values),
            "stdev": statistics.pstdev(utility_values),
        },
        "tie_rate": {
            "best_action_tie_records": tie_records,
            "best_action_tie_rate": tie_records / max(1, total),
            "utility_tie_records": utility_tie_records,
            "utility_tie_rate": utility_tie_rate,
        },
        "zero_sample_weight": {
            "records": zero_weight,
            "rate": zero_weight / max(1, total),
        },
        "state_finite_rate": finite_state / max(1, total),
        "per_sequence_count": dict(sorted(sequence_counts.items())),
        "per_video_count": dict(sorted(video_counts.items(), key=lambda pair: int(pair[0]))),
        "partition": {
            "policy_dataset": str(policy_path),
            "policy_record_count": train_count + val_count,
            "train_records": train_count,
            "val_records": val_count,
            "tracking_dataset": str(tracking_path),
            "tracking_records": test_count,
            "train_sequences": train_sequences,
            "val_sequences": val_sequences,
            "tracking_test_sequences": test_sequences,
            "sequence_intersections": {
                "train_val": sorted(set(train_sequences) & set(val_sequences)),
                "train_tracking": sorted(set(train_sequences) & set(test_sequences)),
                "val_tracking": sorted(set(val_sequences) & set(test_sequences)),
            },
            "split_manifest": str(PILOT / "PILOT_POLICY_SPLIT.json"),
        },
        "suspicious_reasons": suspicious_reasons,
        "go_no_go_for_training": "STOP_AND_REPORT" if suspicious_reasons else "CONTINUE_SCREENING",
    }
    write_json(PILOT / "PILOT_DATA_AUDIT.json", audit)
    write_json(
        PILOT / "PILOT_SNAPSHOT_MANIFEST.json",
        {
            "status": status,
            "created_utc": created,
            "read_only": True,
            "formal_builders_modified": False,
            "sources": sources,
            "snapshot_files": [str(path) for path in sorted(snapshot_root.glob("video_*.jsonl"))],
        },
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("snapshot-audit",), default="snapshot-audit")
    args = parser.parse_args()
    if args.stage == "snapshot-audit":
        result = snapshot_and_audit()
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
