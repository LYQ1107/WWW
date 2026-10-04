"""Decision-policy metrics used by the JEV comparison report."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch


def _aligned_target(record: Mapping[str, Any], prediction: Mapping[str, Any]) -> List[float]:
    target = dict(zip(record["legal_actions"], record["target_probs"]))
    return [float(target[name]) for name in prediction["legal_actions"]]


def brier_score(records: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> float:
    if len(records) != len(predictions) or not records:
        raise ValueError("records and predictions must be non-empty and aligned")
    total = 0.0
    for record, prediction in zip(records, predictions):
        target = _aligned_target(record, prediction)
        probability = [float(value) for value in prediction["probabilities"]]
        if len(target) != len(probability):
            raise ValueError("prediction action set does not match record")
        total += sum((p - y) ** 2 for p, y in zip(probability, target))
    return total / len(records)


def nll(records: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> float:
    if len(records) != len(predictions) or not records:
        raise ValueError("records and predictions must be non-empty and aligned")
    total = 0.0
    for record, prediction in zip(records, predictions):
        target = _aligned_target(record, prediction)
        probability = [max(float(value), 1e-8) for value in prediction["probabilities"]]
        total -= sum(y * math.log(p) for y, p in zip(target, probability))
    return total / len(records)


def expected_calibration_error(
    confidence: Sequence[float], correct: Sequence[float], bins: int = 10
) -> float:
    if len(confidence) != len(correct):
        raise ValueError("confidence and correct must be aligned")
    if not confidence or bins < 1:
        return 0.0
    total = len(confidence)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = [
            i for i, value in enumerate(confidence)
            if lower <= value < upper
            or (index == bins - 1 and lower <= value <= upper)
        ]
        if selected:
            mean_confidence = sum(float(confidence[i]) for i in selected) / len(selected)
            mean_correct = sum(float(correct[i]) for i in selected) / len(selected)
            error += len(selected) / total * abs(mean_confidence - mean_correct)
    return error


def risk_coverage(
    confidence: Sequence[float], correct: Sequence[float],
    coverage_levels: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
) -> List[Dict[str, float]]:
    if len(confidence) != len(correct) or not confidence:
        raise ValueError("confidence and correct must be non-empty and aligned")
    order = sorted(range(len(confidence)), key=lambda index: float(confidence[index]), reverse=True)
    result = []
    for coverage in coverage_levels:
        if not 0 < coverage <= 1:
            raise ValueError("coverage must be in (0, 1]")
        count = max(1, int(math.ceil(len(order) * coverage)))
        selected = order[:count]
        accuracy = sum(float(correct[index]) for index in selected) / count
        result.append({"coverage": float(count / len(order)), "risk": float(1.0 - accuracy)})
    return result


def summarize(
    records: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    confidence = []
    correct = []
    for record, prediction in zip(records, predictions):
        values = [float(value) for value in prediction["probabilities"]]
        index = max(range(len(values)), key=values.__getitem__)
        chosen = prediction["legal_actions"][index]
        confidence.append(values[index])
        correct.append(float(chosen in record["best_actions"]))
    return {
        "records": len(records),
        "nll": nll(records, predictions),
        "brier": brier_score(records, predictions),
        "ece": expected_calibration_error(confidence, correct),
        "risk_coverage": risk_coverage(confidence, correct),
        "best_action_accuracy": sum(correct) / len(correct),
    }
