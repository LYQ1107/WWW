"""Aggregate the three locked methods from the shared compact H=8 arrays."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch

from aggregate_jev_three_way import METHODS, SEEDS, mean_std, write_markdown
from gtr.modeling.jev_runtime import build_controller_from_checkpoint
from jev_compact_dataset import CompactJEVData, OUTCOME_FIELDS
from train_jev import choose_model, load_policy_split


def predict(model, data: CompactJEVData, indices: np.ndarray):
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), 2048):
            group = indices[start : start + 2048]
            features = torch.as_tensor(np.asarray(data.features[group]), dtype=torch.float32)
            questions = torch.as_tensor(np.asarray(data.questions[group]), dtype=torch.long)
            legal = torch.as_tensor(np.asarray(data.legal_actions[group]), dtype=torch.long)
            rows.append(model(features, questions, legal)["probs"].cpu().numpy())
    return np.concatenate(rows, axis=0) if rows else np.empty((0, len(data.manifest["action_names"])), dtype=np.float32)


def validation_metrics(data: CompactJEVData, indices: np.ndarray, probabilities: np.ndarray):
    legal = np.asarray(data.legal_actions[indices])
    target = np.asarray(data.target_probs[indices], dtype=np.float64)
    best = np.asarray(data.best_mask[indices])
    weights = np.asarray(data.sample_weight[indices], dtype=np.float64)
    outcomes = np.asarray(data.outcomes[indices], dtype=np.float64)
    denominator = float(weights.sum())
    nll = brier = correct = 0.0
    confidence = []
    action_distribution: Dict[str, float] = {}
    best_distribution: Dict[str, float] = {}
    mechanism: Dict[str, float] = {}
    for row in range(len(indices)):
        valid = legal[row] >= 0
        probs = probabilities[row]
        chosen_col = int(np.argmax(probs))
        chosen_id = int(legal[row, chosen_col])
        weight = float(weights[row])
        chosen_name = data.manifest["action_names"][chosen_id]
        hit = float(best[row, chosen_id])
        nll += weight * float(-(target[row, valid] * np.log(np.maximum(probs[valid], 1e-8))).sum())
        brier += weight * float(((probs[valid] - target[row, valid]) ** 2).sum())
        correct += weight * hit
        confidence.append((float(probs[chosen_col]), hit, weight))
        action_distribution[chosen_name] = action_distribution.get(chosen_name, 0.0) + weight
        best_ids = np.flatnonzero(best[row])
        if len(best_ids):
            best_name = data.manifest["action_names"][int(best_ids[0])]
            best_distribution[best_name] = best_distribution.get(best_name, 0.0) + weight
        for field_index, field in enumerate(OUTCOME_FIELDS):
            value = outcomes[row, chosen_id, field_index]
            if math.isfinite(float(value)):
                mechanism[field] = mechanism.get(field, 0.0) + weight * float(value)
    ece = 0.0
    for bucket in range(10):
        lower, upper = bucket / 10.0, (bucket + 1) / 10.0
        selected = [item for item in confidence if lower <= item[0] < upper or (bucket == 9 and item[0] <= upper)]
        bucket_weight = sum(item[2] for item in selected)
        if bucket_weight:
            ece += bucket_weight / denominator * abs(
                sum(item[0] * item[2] for item in selected) / bucket_weight
                - sum(item[1] * item[2] for item in selected) / bucket_weight
            )
    oracle = 0.0
    utility_index = OUTCOME_FIELDS.index("utility")
    for row in range(len(indices)):
        valid = legal[row] >= 0
        oracle += weights[row] * float(np.nanmax(outcomes[row, valid, utility_index]))
    return {
        "records": int(len(indices)),
        "weighted_records": denominator,
        "val_utility": mechanism.get("utility", 0.0) / denominator,
        "oracle_best_utility": oracle / denominator,
        "nll": nll / denominator,
        "accuracy": correct / denominator,
        "brier": brier / denominator,
        "ece": ece,
        "action_distribution": {name: value / denominator for name, value in sorted(action_distribution.items())},
        "best_action_distribution": {name: value / denominator for name, value in sorted(best_distribution.items())},
        "mechanism": {field: value / denominator for field, value in sorted(mechanism.items())},
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--methods-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    data = CompactJEVData(args.dataset.resolve())
    split = load_policy_split(args.split_manifest.resolve())
    val_indices = data.indices_for_sequences(split["val"])
    val_indices = val_indices[np.asarray(data.sample_weight[val_indices]) > 0]
    if len(val_indices) == 0:
        raise ValueError("validation split has no positive-weight compact records")

    report_methods: Dict[str, Any] = {}
    for method, label in METHODS.items():
        root = args.methods_root.resolve() / method
        manifest = json.loads((root / "method_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "PASS" or list(manifest.get("seeds", ())) != list(SEEDS):
            raise ValueError(f"method manifest incomplete: {root}")
        seed_metrics = []
        for member in manifest["members"]:
            controller = build_controller_from_checkpoint(member["calibrated_checkpoint"], device="cpu")
            probabilities = predict(controller, data, val_indices)
            metrics = validation_metrics(data, val_indices, probabilities)
            metrics["seed"] = int(member["seed"])
            metrics["temperature"] = float(member["temperature"])
            seed_metrics.append(metrics)
        model = choose_model(method, int(data.manifest["state_dim"]), int(manifest["training"]["hidden_dim"]))
        params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        report_methods[method] = {
            "label": label,
            "hidden_dim": int(manifest["training"]["hidden_dim"]),
            "trainable_params": int(params),
            "seed_metrics": seed_metrics,
        }
    target_params = report_methods["jev"]["trainable_params"]
    scalar_fields = ("val_utility", "oracle_best_utility", "nll", "accuracy", "brier", "ece")
    for item in report_methods.values():
        item["aggregate"] = {
            field: mean_std([float(row[field]) for row in item["seed_metrics"]])
            for field in scalar_fields
        }
        action_names = sorted({action for row in item["seed_metrics"] for action in row["action_distribution"]})
        item["aggregate"]["action_distribution"] = {
            action: mean_std([float(row["action_distribution"].get(action, 0.0)) for row in item["seed_metrics"]])
            for action in action_names
        }
        mechanism_names = sorted({metric for row in item["seed_metrics"] for metric in row["mechanism"]})
        item["aggregate"]["mechanism"] = {
            metric: mean_std([float(row["mechanism"].get(metric, 0.0)) for row in item["seed_metrics"]])
            for metric in mechanism_names
        }
        item["relative_param_difference"] = abs(item["trainable_params"] - target_params) / max(1, target_params)
        if item["relative_param_difference"] >= 0.02:
            raise ValueError(f"parameter matching gate failed for {item['label']}")
    jev_utility = report_methods["jev"]["aggregate"]["val_utility"]["mean"]
    manifest_path = args.dataset.resolve() / "manifest.json"
    report = {
        "status": "PASS",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(manifest_path),
        "policy_split": str(args.split_manifest.resolve()),
        "policy_split_sha256": sha256(args.split_manifest.resolve()),
        "horizon": 8,
        "state_dim": int(data.manifest["state_dim"]),
        "seeds": list(SEEDS),
        "training": {"epochs": 50, "batch_size": 128, "optimizer": "AdamW", "learning_rate": 1e-3, "calibration": "temperature_scaling_policy_val_only"},
        "methods": report_methods,
        "jev_beats_both_on_val_utility": bool(
            jev_utility > report_methods["question_threshold"]["aggregate"]["val_utility"]["mean"]
            and jev_utility > report_methods["question_conditioned_mlp"]["aggregate"]["val_utility"]["mean"]
        ),
        "official_test_read": False,
        "tracking_gate": "run Frozen GMT + three methods only if JEV mean validation utility beats both controls",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, args.markdown)
    print(json.dumps({"status": "PASS", "output": str(args.output), "jev_beats_both_on_val_utility": report["jev_beats_both_on_val_utility"]}, indent=2))


if __name__ == "__main__":
    main()
