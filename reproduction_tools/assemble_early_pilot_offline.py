"""Assemble single-seed offline metrics for the early H=8 pilot."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
PILOT = Path("/home/liuyeqiang/WWW_jev_full_h8_runtime/pilot")
METHODS = {
    "question_threshold": "Learnable Threshold",
    "question_conditioned_mlp": "Generic MLP",
    "jev": "Full JEV",
}
SEED = 20261003


def json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_modules():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "third_party" / "CenterNet2"))
    sys.path.insert(0, str(ROOT / "reproduction_tools"))
    from aggregate_jev_three_way import predictions_for, weighted_metrics
    from gtr.modeling.jev_runtime import build_controller_from_checkpoint
    from train_jev import read_records

    return predictions_for, weighted_metrics, build_controller_from_checkpoint, read_records


def baselines(train: Sequence[Mapping[str, Any]], val: Sequence[Mapping[str, Any]]):
    mode_by_question: Dict[str, str] = {}
    for question in {str(record["question_type"]) for record in train}:
        values = [str(record["best_actions"][0]) for record in train if record["question_type"] == question]
        mode_by_question[question] = Counter(values).most_common(1)[0][0]

    def score(policy):
        total_weight = 0.0
        correct = 0.0
        utility = 0.0
        expected_utility = 0.0
        for record in val:
            weight = float(record.get("sample_weight", 1.0))
            if weight <= 0:
                continue
            legal = list(record["legal_actions"])
            action = policy(record)
            if action not in legal:
                action = legal[0]
            outcome = record["action_outcomes"][action]
            correct += weight * float(action in record["best_actions"])
            utility += weight * float(outcome["utility"])
            expected_utility += weight * sum(
                float(item["utility"]) for item in record["action_outcomes"].values()
            ) / len(record["action_outcomes"])
            total_weight += weight
        return {
            "records": len(val),
            "weighted_records": total_weight,
            "best_action_accuracy": correct / max(1e-8, total_weight),
            "validation_utility": utility / max(1e-8, total_weight),
            "random_uniform_expected_utility": expected_utility / max(1e-8, total_weight),
        }

    majority = score(lambda record: mode_by_question[str(record["question_type"])])
    random_accuracy = 0.0
    random_utility = 0.0
    weight_total = 0.0
    for record in val:
        weight = float(record.get("sample_weight", 1.0))
        if weight <= 0:
            continue
        legal = list(record["legal_actions"])
        random_accuracy += weight * len(set(record["best_actions"]) & set(legal)) / len(legal)
        random_utility += weight * sum(
            float(record["action_outcomes"][action]["utility"]) for action in legal
        ) / len(legal)
        weight_total += weight
    random_baseline = {
        "records": len(val),
        "weighted_records": weight_total,
        "best_action_accuracy": random_accuracy / max(1e-8, weight_total),
        "validation_utility": random_utility / max(1e-8, weight_total),
        "policy": "uniform over legal actions",
    }
    return {"majority_action": majority, "random_uniform": random_baseline, "majority_by_question": mode_by_question}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    dataset = PILOT / "PILOT_POLICY.jsonl"
    split_path = PILOT / "PILOT_POLICY_SPLIT.json"
    if not dataset.is_file() or not split_path.is_file():
        raise FileNotFoundError("pilot policy dataset/split missing")
    predictions_for, weighted_metrics, build_controller_from_checkpoint, read_records = load_modules()
    records = read_records(dataset)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    val_names = set(str(value) for value in split["val_sequences"])
    train_names = set(str(value) for value in split["train_sequences"])
    train = [record for record in records if str(record["sequence"]) in train_names]
    val = [record for record in records if str(record["sequence"]) in val_names]
    report_methods: Dict[str, Any] = {}
    pending = []
    for method, label in METHODS.items():
        root = PILOT / "offline" / method
        metrics_path = root / "metrics.json"
        checkpoint = root / "model.pth"
        if not metrics_path.is_file() or not checkpoint.is_file():
            pending.append(method)
            continue
        train_report = json.loads(metrics_path.read_text(encoding="utf-8"))
        controller = build_controller_from_checkpoint(checkpoint, device="cpu")
        predictions = predictions_for(controller, val)
        metrics = weighted_metrics(val, predictions)
        report_methods[method] = {
            "label": label,
            "checkpoint": str(checkpoint),
            "seed": SEED,
            "training": {
                "epochs": len(train_report.get("history", [])),
                "batch_size": 256,
                "optimizer": "AdamW",
                "learning_rate": 0.001,
                "device": "cpu",
                "loss_curve": [
                    {"epoch": item["epoch"], "train_loss": item["train_loss"], "val_nll": item["val"].get("nll")}
                    for item in train_report.get("history", [])
                ],
            },
            "validation": metrics,
        }
    if pending:
        report = {
            "status": "TRAINING_PENDING",
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "classification": "SCREENING_ONLY_NOT_FOR_FINAL_SELECTION_NOT_FOR_PAPER_RESULT",
            "pending_methods": pending,
            "methods": report_methods,
        }
        json_write(PILOT / "PILOT_OFFLINE_THREE_WAY.json", report)
        print(json.dumps(report, indent=2))
        raise SystemExit(2)
    report = {
        "status": "PASS",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": "SCREENING_ONLY_NOT_FOR_FINAL_SELECTION_NOT_FOR_PAPER_RESULT",
        "seed": SEED,
        "dataset": str(dataset),
        "policy_split": str(split_path),
        "train_records": len(train),
        "val_records": len(val),
        "official_test_read": False,
        "methods": report_methods,
        "baselines": baselines(train, val),
        "gate": {
            "jev_beats_mlp_on_validation_utility": report_methods["jev"]["validation"]["val_utility"] > report_methods["question_conditioned_mlp"]["validation"]["val_utility"],
            "jev_beats_threshold_on_validation_utility": report_methods["jev"]["validation"]["val_utility"] > report_methods["question_threshold"]["validation"]["val_utility"],
        },
    }
    report["gate"]["pilot_offline_signal"] = bool(
        report["gate"]["jev_beats_mlp_on_validation_utility"]
        and report["gate"]["jev_beats_threshold_on_validation_utility"]
    )
    json_write(PILOT / "PILOT_OFFLINE_THREE_WAY.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
