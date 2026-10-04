"""Offline decision/stress diagnostics for a frozen JEV dataset.

The input is an explicitly future-GT-labelled counterfactual dataset.  This
tool never claims an online tracking result: it measures whether states with a
near-identical legacy score can have different long-horizon action targets,
and compares the trained policy distributions on the same records.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

from jev_metrics import summarize
from train_jev import read_records
from gtr.modeling.jev_runtime import build_controller_from_checkpoint


SCORE_INDEX = {
    "MATCH_DECISION": 0,
    "MEMORY_DECISION": 2,
    "REACTIVATION_DECISION": 3,
}


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def predictions_for(model: torch.nn.Module, records: Sequence[Mapping[str, Any]]):
    """Evaluate records in groups with the same typed legal action set."""
    groups: Dict[Tuple[str, Tuple[str, ...]], List[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[(str(record["question_type"]), tuple(record["legal_actions"]))].append(index)
    predictions: List[Dict[str, Any] | None] = [None] * len(records)
    model.eval()
    with torch.no_grad():
        for (question, legal), indices in groups.items():
            features = torch.tensor(
                [records[index]["state"]["feature_vector"] for index in indices],
                dtype=torch.float32,
            )
            output = model(features, [question] * len(indices), [list(legal)] * len(indices))
            for row, index in enumerate(indices):
                values = output["probs"][row, : len(legal)].detach().cpu().tolist()
                predictions[index] = {
                    "legal_actions": list(legal),
                    "probabilities": [float(value) for value in values],
                }
    if any(value is None for value in predictions):
        raise AssertionError("missing policy prediction")
    return [value for value in predictions if value is not None]


def same_score_pairs(
    records: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    epsilon: float,
    max_pairs: int,
) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    for index, record in enumerate(records):
        question = str(record["question_type"])
        feature_index = SCORE_INDEX[question]
        score = float(record["state"]["feature_vector"][feature_index])
        grouped[(question, int(round(score / epsilon)))].append(index)

    pairs: List[Dict[str, Any]] = []
    by_question: Dict[str, Dict[str, Any]] = {
        question: {
            "records": 0,
            "candidate_pairs": 0,
            "oracle_action_disagreement_pairs": 0,
            "policy_action_disagreement_pairs": defaultdict(int),
            "max_score_gap": 0.0,
            "mean_state_l2": 0.0,
        }
        for question in SCORE_INDEX
    }
    for (question, _bucket), indices in grouped.items():
        by_question[question]["records"] += len(indices)
        for left_pos in range(len(indices)):
            if len(pairs) >= max_pairs:
                break
            left = indices[left_pos]
            left_record = records[left]
            left_features = left_record["state"]["feature_vector"]
            left_best = set(left_record["best_actions"])
            for right in indices[left_pos + 1 :]:
                left_score = float(left_features[SCORE_INDEX[question]])
                right_features = records[right]["state"]["feature_vector"]
                right_score = float(right_features[SCORE_INDEX[question]])
                gap = abs(left_score - right_score)
                distance = math.sqrt(
                    sum(
                        (float(a) - float(b)) ** 2
                        for a, b in zip(left_features, right_features)
                    )
                )
                by_question[question]["candidate_pairs"] += 1
                by_question[question]["max_score_gap"] = max(
                    by_question[question]["max_score_gap"], gap
                )
                if distance < 0.25 or not left_best.isdisjoint(set(records[right]["best_actions"])):
                    continue
                by_question[question]["oracle_action_disagreement_pairs"] += 1
                by_question[question]["mean_state_l2"] += distance
                row = {
                    "question": question,
                    "left_index": left,
                    "right_index": right,
                    "score_gap": gap,
                    "state_l2": distance,
                    "left_best_actions": sorted(left_best),
                    "right_best_actions": sorted(set(records[right]["best_actions"])),
                }
                for name, policy_predictions in predictions.items():
                    left_prediction = policy_predictions[left]
                    right_prediction = policy_predictions[right]
                    left_choice = left_prediction["legal_actions"][max(range(len(left_prediction["probabilities"])), key=left_prediction["probabilities"].__getitem__)]
                    right_choice = right_prediction["legal_actions"][max(range(len(right_prediction["probabilities"])), key=right_prediction["probabilities"].__getitem__)]
                    differs = left_choice != right_choice
                    by_question[question]["policy_action_disagreement_pairs"][name] += int(differs)
                    row.setdefault("policy_choices", {})[name] = {
                        "left": left_choice,
                        "right": right_choice,
                        "different": differs,
                    }
                pairs.append(row)
                if len(pairs) >= max_pairs:
                    break
            if len(pairs) >= max_pairs:
                break

    for values in by_question.values():
        count = int(values["oracle_action_disagreement_pairs"])
        values["mean_state_l2"] = float(values["mean_state_l2"] / count) if count else 0.0
        values["policy_action_disagreement_pairs"] = dict(values["policy_action_disagreement_pairs"])
    return {
        "epsilon": float(epsilon),
        "state_l2_minimum": 0.25,
        "max_pairs": int(max_pairs),
        "pairs_returned": len(pairs),
        "by_question": by_question,
        "examples": pairs[: min(100, len(pairs))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--max-pairs", type=int, default=10000)
    parser.add_argument("policies", nargs="+", help="NAME=POLICY_DIRECTORY")
    args = parser.parse_args()
    if args.epsilon <= 0 or args.max_pairs < 1:
        raise ValueError("epsilon and max-pairs must be positive")
    records = read_records(args.dataset.resolve())
    policy_predictions: Dict[str, Sequence[Mapping[str, Any]]] = {}
    policy_summary: Dict[str, Any] = {}
    for item in args.policies:
        if "=" not in item:
            raise ValueError(f"policy must be NAME=DIR: {item}")
        name, directory = item.split("=", 1)
        if not name:
            raise ValueError("empty policy name")
        checkpoint = Path(directory).resolve() / "model.pth"
        model = build_controller_from_checkpoint(checkpoint, device="cpu")
        predictions = predictions_for(model, records)
        policy_predictions[name] = predictions
        policy_summary[name] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest(),
            "parameters": parameter_count(model),
            "metrics": summarize(records, predictions),
        }
    report = {
        "status": "PASS",
        "dataset": str(args.dataset.resolve()),
        "records": len(records),
        "uses_future_gt": True,
        "diagnostic_scope": "offline counterfactual records only; no online tracking claim",
        "policies": policy_summary,
        "same_score_different_state": same_score_pairs(
            records, policy_predictions, args.epsilon, args.max_pairs
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
