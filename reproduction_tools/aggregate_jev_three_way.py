"""Aggregate the locked H=8 Learnable Threshold/MLP/JEV validation runs."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch

from gtr.modeling.jev_runtime import build_controller_from_checkpoint
from jev_dataset_contract import validate_record
from train_jev import choose_model, load_policy_split, read_records


SEEDS = (20261003, 20261004, 20261005)
METHODS = {
    "question_threshold": "Learnable Threshold",
    "question_conditioned_mlp": "Generic MLP",
    "jev": "Full JEV",
}
MECHANISM_FIELDS = (
    "utility",
    "future_correct_identity_duration",
    "future_identity_switches",
    "future_fragmentation",
    "future_fragments",
    "future_collisions",
    "memory_contamination",
    "contamination_duration",
    "recovery_latency",
    "false_reactivation",
    "new_id_fragmentation",
    "window_idf1",
    "window_assa_proxy",
    "future_events",
    "informative",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        raise ValueError("cannot aggregate empty values")
    mean = sum(float(value) for value in values) / len(values)
    if len(values) < 2:
        return {"mean": mean, "std": 0.0}
    variance = sum((float(value) - mean) ** 2 for value in values) / (len(values) - 1)
    return {"mean": mean, "std": math.sqrt(max(0.0, variance))}


def predictions_for(model, records: Sequence[Mapping[str, Any]], batch_size: int = 2048):
    groups: Dict[tuple[str, tuple[str, ...]], List[int]] = {}
    for index, record in enumerate(records):
        key = (str(record["question_type"]), tuple(record["legal_actions"]))
        groups.setdefault(key, []).append(index)
    predictions: List[Dict[str, Any] | None] = [None] * len(records)
    model.eval()
    with torch.no_grad():
        for (question, legal), indices in groups.items():
            for start in range(0, len(indices), batch_size):
                selected = indices[start : start + batch_size]
                features = torch.tensor(
                    [records[index]["state"]["feature_vector"] for index in selected],
                    dtype=torch.float32,
                )
                output = model(features, [question] * len(selected), [list(legal)] * len(selected))
                for row, index in enumerate(selected):
                    values = output["probs"][row, : len(legal)].detach().cpu().tolist()
                    predictions[index] = {
                        "legal_actions": list(legal),
                        "probabilities": [float(value) for value in values],
                    }
    if any(value is None for value in predictions):
        raise AssertionError("missing policy prediction")
    return [value for value in predictions if value is not None]


def weighted_metrics(records: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]):
    if len(records) != len(predictions) or not records:
        raise ValueError("records and predictions must be non-empty and aligned")
    denominator = sum(float(record.get("sample_weight", 1.0)) for record in records)
    if denominator <= 0:
        raise ValueError("validation has no positive sample weight")
    nll = brier = correct = 0.0
    confidence: List[tuple[float, float, float]] = []
    action_weights: Dict[str, float] = {}
    best_weights: Dict[str, float] = {}
    mechanism_totals: Dict[str, float] = {}
    oracle_utility = 0.0
    for record, prediction in zip(records, predictions):
        weight = float(record.get("sample_weight", 1.0))
        target = dict(zip(record["legal_actions"], record["target_probs"]))
        probs = dict(zip(prediction["legal_actions"], prediction["probabilities"]))
        nll += weight * sum(-float(target[name]) * math.log(max(float(probs[name]), 1e-8)) for name in probs)
        brier += weight * sum((float(probs[name]) - float(target[name])) ** 2 for name in probs)
        chosen = max(prediction["legal_actions"], key=lambda name: probs[name])
        hit = float(chosen in record["best_actions"])
        correct += weight * hit
        confidence.append((float(probs[chosen]), hit, weight))
        action_weights[chosen] = action_weights.get(chosen, 0.0) + weight
        best = str(record["best_actions"][0])
        best_weights[best] = best_weights.get(best, 0.0) + weight
        outcome = record["action_outcomes"][chosen]
        for field in MECHANISM_FIELDS:
            value = outcome.get(field)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                mechanism_totals[field] = mechanism_totals.get(field, 0.0) + weight * float(value)
        oracle_utility += weight * max(float(item["utility"]) for item in record["action_outcomes"].values())

    ece = 0.0
    for lower_index in range(10):
        lower = lower_index / 10.0
        upper = (lower_index + 1) / 10.0
        bucket = [item for item in confidence if lower <= item[0] < upper or (lower_index == 9 and item[0] <= upper)]
        bucket_weight = sum(item[2] for item in bucket)
        if bucket_weight:
            ece += bucket_weight / denominator * abs(
                sum(item[0] * item[2] for item in bucket) / bucket_weight
                - sum(item[1] * item[2] for item in bucket) / bucket_weight
            )
    return {
        "records": len(records),
        "weighted_records": denominator,
        "val_utility": mechanism_totals.get("utility", 0.0) / denominator,
        "oracle_best_utility": oracle_utility / denominator,
        "nll": nll / denominator,
        "accuracy": correct / denominator,
        "brier": brier / denominator,
        "ece": ece,
        "action_distribution": {name: value / denominator for name, value in sorted(action_weights.items())},
        "best_action_distribution": {name: value / denominator for name, value in sorted(best_weights.items())},
        "mechanism": {
            field: total / denominator for field, total in sorted(mechanism_totals.items())
        },
    }


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    methods = report["methods"]
    lines = [
        "# H=8 JEV Three-Way Validation",
        "",
        "This report compares Learnable Threshold, Generic MLP, and Full JEV on one shared canonical TRAIN H=8 counterfactual dataset.",
        "Official TEST data is not read for this protocol.",
        "",
        "## Locked protocol",
        "",
        f"- Seeds: `{', '.join(str(seed) for seed in report['seeds'])}`; split is sequence-disjoint and fixed before training.",
        f"- Training: `{report['training']['epochs']}` epochs, batch `{report['training']['batch_size']}`, AdamW, LR `{report['training']['learning_rate']}`.",
        "- All methods consume identical state vectors, typed questions, legal actions, target probabilities, and sample weights.",
        "- Temperature scaling is fitted on policy validation only and applied to all three method families.",
        "",
        "## Main validation metrics (mean ± sample std over seeds)",
        "",
        "| Method | Params | Val Utility | NLL | Accuracy | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in methods.items():
        agg = item["aggregate"]
        lines.append(
            f"| {item['label']} | {item['trainable_params']:,} | "
            f"{agg['val_utility']['mean']:.6f} ± {agg['val_utility']['std']:.6f} | "
            f"{agg['nll']['mean']:.6f} ± {agg['nll']['std']:.6f} | "
            f"{agg['accuracy']['mean']:.6f} ± {agg['accuracy']['std']:.6f} | "
            f"{agg['brier']['mean']:.6f} ± {agg['brier']['std']:.6f} | "
            f"{agg['ece']['mean']:.6f} ± {agg['ece']['std']:.6f} |"
        )
    lines += [
        "",
        "## Parameter matching",
        "",
        f"Full JEV is the parameter target ({methods['jev']['trainable_params']:,}). "
        f"Learnable Threshold differs by {methods['question_threshold']['relative_param_difference']:.2%}; "
        f"Generic MLP differs by {methods['question_conditioned_mlp']['relative_param_difference']:.2%}.",
        "",
        "## Action distribution and mechanism metrics",
        "",
    ]
    for key, item in methods.items():
        lines.append(f"### {item['label']}")
        lines.append("")
        lines.append("Action distribution (mean ± std):")
        for action, value in item["aggregate"]["action_distribution"].items():
            lines.append(f"- `{action}`: {value['mean']:.4f} ± {value['std']:.4f}")
        lines.append("")
        lines.append("Selected-action mechanism metrics (mean ± std):")
        for metric, value in item["aggregate"]["mechanism"].items():
            lines.append(f"- `{metric}`: {value['mean']:.6f} ± {value['std']:.6f}")
        lines.append("")
    lines += [
        "## Gate for expensive tracking evaluation",
        "",
        f"`jev_beats_both_on_val_utility = {report['jev_beats_both_on_val_utility']}`.",
        "Tracking head-to-head is authorized only when this gate is true; otherwise the protocol stops with the validation result.",
        "",
        "## Reproducibility",
        "",
        f"- Shared dataset: `{report['dataset']}` (SHA-256 `{report['dataset_sha256']}`).",
        f"- Policy split: `{report['policy_split']}` (SHA-256 `{report['policy_split_sha256']}`).",
        f"- Generated UTC: `{report['created_utc']}`.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--methods-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    split = args.split_manifest.resolve()
    methods_root = args.methods_root.resolve()
    records = read_records(dataset)
    split_payload = load_policy_split(split)
    val_names = set(split_payload["val"])
    val = [record for record in records if record["sequence"] in val_names and float(record.get("sample_weight", 1.0)) > 0]
    if not val:
        raise ValueError("policy validation split has no positive-weight records")
    dimensions = {len(record["state"]["feature_vector"]) for record in records}
    if len(dimensions) != 1:
        raise ValueError(f"shared dataset has inconsistent feature dimensions: {dimensions}")
    state_dim = next(iter(dimensions))

    report_methods: Dict[str, Any] = {}
    for method, label in METHODS.items():
        manifest_path = methods_root / method / "method_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "PASS" or list(manifest.get("seeds", ())) != list(SEEDS):
            raise ValueError(f"method manifest is incomplete: {manifest_path}")
        seed_metrics = []
        for member in manifest["members"]:
            calibrated = Path(member["calibrated_checkpoint"])
            controller = build_controller_from_checkpoint(calibrated, device="cpu")
            prediction = predictions_for(controller, val)
            metrics = weighted_metrics(val, prediction)
            metrics["seed"] = int(member["seed"])
            metrics["temperature"] = float(member["temperature"])
            seed_metrics.append(metrics)
        model = choose_model(method, state_dim, int(manifest["training"]["hidden_dim"]))
        params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        report_methods[method] = {
            "label": label,
            "hidden_dim": int(manifest["training"]["hidden_dim"]),
            "trainable_params": int(params),
            "seed_metrics": seed_metrics,
        }

    target_params = report_methods["jev"]["trainable_params"]
    scalar_fields = (
        "val_utility",
        "oracle_best_utility",
        "nll",
        "accuracy",
        "brier",
        "ece",
    )
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
    report = {
        "status": "PASS",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset),
        "policy_split": str(split),
        "policy_split_sha256": sha256(split),
        "horizon": 8,
        "state_dim": state_dim,
        "seeds": list(SEEDS),
        "training": {
            "epochs": 50,
            "batch_size": 128,
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "calibration": "temperature_scaling_policy_val_only",
        },
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
