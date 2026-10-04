"""Offline helpers for reproducible JEV counterfactual records.

This module contains labeler-side bookkeeping only.  It can construct a
future-GT record, but no online model should import it; the strict JSONL
validator still rejects such a record unless ``allow_future_gt=True``.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from gtr.modeling.jev_decision import ACTION_NAMES, QUESTION_NAMES, action_index, question_index

from jev_dataset_contract import digest_state, validate_record


TIE_BREAK: Mapping[str, Tuple[str, ...]] = {
    "MATCH_DECISION": ("ACCEPT_CURRENT", "REASSOCIATE", "START_NEW"),
    "MEMORY_DECISION": ("SKIP_MEMORY", "WRITE_MEMORY"),
    "REACTIVATION_DECISION": ("REACTIVATE_OLD", "START_NEW"),
}


def scalar_utility(outcome: Any) -> float:
    """Extract a finite scalar utility from a branch outcome."""

    value = outcome.get("utility") if isinstance(outcome, Mapping) else outcome
    if isinstance(value, Mapping):
        raise ValueError("branch outcome utility must be scalar after weighting")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("utility must be finite")
    return value


def best_actions(
    question: str,
    legal_actions: Sequence[str],
    utilities: Mapping[str, float],
    *,
    tie_tolerance: float = 1e-8,
) -> Tuple[str, ...]:
    """Return all max-utility actions using a deterministic semantic order."""

    question_index(question)
    if not legal_actions or set(legal_actions) != set(utilities):
        raise ValueError("utilities must match legal_actions exactly")
    values = {name: scalar_utility(value) for name, value in utilities.items()}
    maximum = max(values.values())
    ordered = [name for name in TIE_BREAK[question] if name in values]
    ordered.extend(name for name in legal_actions if name not in ordered)
    return tuple(
        name for name in ordered if maximum - values[name] <= float(tie_tolerance)
    )


def utility_target(
    question: str,
    legal_actions: Sequence[str],
    utilities: Mapping[str, float],
    *,
    temperature: float = 1.0,
    tie_tolerance: float = 1e-8,
) -> Tuple[Tuple[str, ...], Tuple[float, ...]]:
    """Build deterministic best-action and normalized soft utility targets."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    best = best_actions(question, legal_actions, utilities, tie_tolerance=tie_tolerance)
    values = [scalar_utility(utilities[name]) for name in legal_actions]
    maximum = max(values)
    weights = [math.exp((value - maximum) / temperature) for value in values]
    normalizer = sum(weights)
    probs = tuple(weight / normalizer for weight in weights)
    return best, probs


def make_record(
    *,
    dataset: str,
    sequence: str,
    frame: int,
    view: int,
    question_type: str,
    state: Mapping[str, Any],
    legal_actions: Sequence[str],
    action_outcomes: Mapping[str, Mapping[str, Any]],
    gmt_checkpoint_sha256: str,
    horizon: int,
    temperature: float = 1.0,
    tie_tolerance: float = 1e-8,
) -> Dict[str, Any]:
    """Construct and validate one offline future-GT record."""

    question_index(question_type)
    legal = list(legal_actions)
    if set(action_outcomes) != set(legal):
        raise ValueError("action_outcomes must match legal_actions exactly")
    utilities = {name: scalar_utility(action_outcomes[name]) for name in legal}
    weights = [
        float(action_outcomes[name].get("sample_weight", 1.0))
        if isinstance(action_outcomes[name], Mapping)
        else 1.0
        for name in legal
    ]
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("action sample_weight values must be finite and non-negative")
    if max(weights) - min(weights) > 1e-8:
        raise ValueError("all actions in one decision must share sample_weight")
    sample_weight = weights[0]
    best, target_probs = utility_target(
        question_type,
        legal,
        utilities,
        temperature=temperature,
        tie_tolerance=tie_tolerance,
    )
    record = {
        "schema_version": 1,
        "dataset": dataset,
        "sequence": sequence,
        "frame": int(frame),
        "view": int(view),
        "question_type": question_type,
        "state_digest": digest_state(state),
        "gmt_checkpoint_sha256": gmt_checkpoint_sha256,
        "legal_actions": legal,
        "state": dict(state),
        "action_outcomes": dict(action_outcomes),
        "best_actions": list(best),
        "target_probs": list(target_probs),
        "horizon": int(horizon),
        "uses_future_gt": True,
        "sample_weight": sample_weight,
        "labeling": {
            "temperature": float(temperature),
            "tie_tolerance": float(tie_tolerance),
            "tie_break_order": list(TIE_BREAK[question_type]),
            "sample_weight": sample_weight,
        },
    }
    validate_record(record, allow_future_gt=True)
    return record


def split_sequences(
    sequences: Iterable[str],
    *,
    seed: int = 20261003,
    fractions: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> Dict[str, Tuple[str, ...]]:
    """Split by sequence name, never by adjacent frame."""

    if len(fractions) != 3 or any(value <= 0 for value in fractions):
        raise ValueError("fractions must contain three positive values")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("fractions must sum to one")
    unique = sorted(set(str(value) for value in sequences))
    shuffled = list(unique)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, int(round(n * fractions[0]))) if n else 0
    n_val = max(1, int(round(n * fractions[1]))) if n >= 3 else 0
    if n_train + n_val >= n and n >= 3:
        n_train = max(1, n - 2)
        n_val = 1
    result = {
        "train": tuple(sorted(shuffled[:n_train])),
        "val": tuple(sorted(shuffled[n_train : n_train + n_val])),
        "test": tuple(sorted(shuffled[n_train + n_val :])),
    }
    if set(result["train"]) & set(result["val"]):
        raise AssertionError("sequence split overlap")
    if set(result["train"]) & set(result["test"]):
        raise AssertionError("sequence split overlap")
    if set(result["val"]) & set(result["test"]):
        raise AssertionError("sequence split overlap")
    return result
