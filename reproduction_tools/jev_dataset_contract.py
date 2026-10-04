"""Validation helpers for the offline JEV counterfactual JSONL contract.

The validator is intentionally strict about the boundary between online state
and future-GT labels.  It is usable by both a future labeler and online
consumers; online consumers must call ``validate_record(...,
allow_future_gt=False)``.
"""

import hashlib
import json
import math
from typing import Any, Dict, Iterable, Mapping, Sequence

from gtr.modeling.jev_decision import (
    ACTION_NAMES,
    QUESTION_NAMES,
    action_index,
    question_index,
    _validate_actions,
)


SCHEMA_VERSION = 1
REQUIRED_FIELDS = {
    "schema_version",
    "dataset",
    "sequence",
    "frame",
    "view",
    "question_type",
    "state_digest",
    "gmt_checkpoint_sha256",
    "legal_actions",
    "state",
    "action_outcomes",
    "best_actions",
    "target_probs",
    "horizon",
    "uses_future_gt",
}
FORBIDDEN_ONLINE_KEYS = {
    "gt",
    "ground_truth",
    "future_gt",
    "future_ground_truth",
    "evaluator_feedback",
    "future_metrics",
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def digest_state(state: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(state)).hexdigest()


def _walk_forbidden(value: Any, path: str = "state") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_ONLINE_KEYS:
                raise ValueError(f"future-GT/evaluator field is not allowed at {path}.{key}")
            _walk_forbidden(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _walk_forbidden(child, f"{path}[{index}]")


def validate_record(record: Mapping[str, Any], *, allow_future_gt: bool = False) -> None:
    missing = REQUIRED_FIELDS - set(record)
    if missing:
        raise ValueError(f"missing JEV record fields: {sorted(missing)}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {record['schema_version']}")
    if "sample_weight" in record:
        weight = float(record["sample_weight"])
        if weight < 0 or not math.isfinite(weight):
            raise ValueError("sample_weight must be finite and non-negative")
    question_id = question_index(record["question_type"])
    legal = list(record["legal_actions"])
    if not legal:
        raise ValueError("legal_actions cannot be empty")
    legal_ids = [action_index(action) for action in legal]
    if len(set(legal_ids)) != len(legal_ids):
        raise ValueError("legal_actions contains duplicates")
    _validate_actions(question_id, legal_ids)
    outcomes = record["action_outcomes"]
    if set(outcomes) != set(legal):
        raise ValueError("action_outcomes must match legal_actions exactly")
    best = list(record["best_actions"])
    if not best or not set(best).issubset(set(legal)):
        raise ValueError("best_actions must be a nonempty subset of legal_actions")
    target_probs = list(record["target_probs"])
    if len(target_probs) != len(legal):
        raise ValueError("target_probs must align with legal_actions")
    if any(float(value) < 0 for value in target_probs):
        raise ValueError("target_probs cannot contain negative values")
    if abs(sum(float(value) for value in target_probs) - 1.0) > 1e-4:
        raise ValueError("target_probs must sum to one")
    if int(record["horizon"]) < 1:
        raise ValueError("horizon must be positive")
    if not isinstance(record["uses_future_gt"], bool):
        raise ValueError("uses_future_gt must be boolean")
    if not allow_future_gt and record["uses_future_gt"]:
        raise ValueError("online consumer cannot accept uses_future_gt=true")
    _walk_forbidden(record["state"])

    expected_digest = digest_state(record["state"])
    if record["state_digest"] != expected_digest:
        raise ValueError("state_digest does not match canonical state")


def validate_jsonl(path: str, *, allow_future_gt: bool = False) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                validate_record(record, allow_future_gt=allow_future_gt)
            except Exception as exc:
                raise ValueError(f"invalid JEV record at line {line_number}: {exc}") from exc
            count += 1
    return count
