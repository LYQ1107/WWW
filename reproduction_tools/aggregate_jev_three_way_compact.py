"""Aggregate the three locked methods from the shared compact H=8 arrays."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch

from aggregate_jev_three_way import METHODS
from gtr.modeling.jev_runtime import build_controller_from_checkpoint
from jev_compact_dataset import CompactJEVData, OUTCOME_FIELDS
from train_jev import choose_model, load_policy_split


SEEDS = (20261003,)


def _assert_probability_contract(
    probabilities: np.ndarray,
    legal: np.ndarray,
    *,
    tolerance: float = 1e-6,
) -> None:
    """Fail closed if a policy emits an invalid masked distribution."""

    values = np.asarray(probabilities, dtype=np.float64)
    legal_values = np.asarray(legal, dtype=np.int64)
    if values.ndim != 2 or legal_values.shape != values.shape:
        raise AssertionError(
            f"probability/legal shape mismatch: {values.shape} != {legal_values.shape}"
        )
    if not np.isfinite(values).all():
        raise AssertionError("policy probabilities contain non-finite values")
    if float(values.min(initial=0.0)) < -tolerance or float(values.max(initial=0.0)) > 1.0 + tolerance:
        raise AssertionError("policy probabilities are outside [0, 1]")
    valid = legal_values >= 0
    if not valid.any(axis=1).all():
        raise AssertionError("a validation record has no legal action")
    legal_sums = np.where(valid, values, 0.0).sum(axis=1)
    if not np.all(np.abs(legal_sums - 1.0) <= tolerance):
        raise AssertionError("legal-action probabilities do not sum to one")
    if np.any(np.abs(values[~valid]) > tolerance):
        raise AssertionError("masked illegal-action probabilities are non-zero")


def _ece(confidence: Sequence[tuple[float, float, float]], denominator: float) -> float:
    value = 0.0
    for bucket in range(10):
        lower, upper = bucket / 10.0, (bucket + 1) / 10.0
        selected = [
            item
            for item in confidence
            if lower <= item[0] < upper
            or (lower <= item[0] <= upper and bucket == 9)
        ]
        bucket_weight = sum(item[2] for item in selected)
        if bucket_weight:
            value += bucket_weight / denominator * abs(
                sum(item[0] * item[2] for item in selected) / bucket_weight
                - sum(item[1] * item[2] for item in selected) / bucket_weight
            )
    if not math.isfinite(value) or not (-1e-12 <= value <= 1.0 + 1e-12):
        raise AssertionError(f"ECE is outside [0, 1]: {value}")
    return float(min(1.0, max(0.0, value)))


def _summary(values: Sequence[float]) -> Dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "median": None,
            "p10": None,
            "p50": None,
            "p90": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
    }


def utility_tie_margin_diagnostics(
    data: CompactJEVData,
    indices: np.ndarray,
    *,
    tolerance: float = 1e-8,
) -> Dict[str, Any]:
    """Describe whether validation labels contain a real decision margin."""

    utility_index = OUTCOME_FIELDS.index("utility")
    question_names = data.manifest["question_names"]
    # Keep the full margin values so percentile diagnostics are not reduced to
    # a boolean tie/non-tie flag.
    detailed: Dict[str, list[tuple[float, bool, float]]] = {}
    for index in np.asarray(indices, dtype=np.int64):
        weight = float(data.sample_weight[index])
        if not math.isfinite(weight) or weight < 0:
            raise AssertionError("sample_weight must be finite and non-negative")
        if weight <= 0:
            continue
        question = question_names[int(data.questions[index])]
        legal_ids = np.asarray(data.legal_actions[index], dtype=np.int64)
        legal_ids = legal_ids[legal_ids >= 0]
        utilities = np.asarray(data.outcomes[index], dtype=np.float64)[legal_ids, utility_index]
        if not np.isfinite(utilities).all():
            raise AssertionError("legal action utilities contain non-finite values")
        best_value = float(np.max(utilities))
        best_count = int(np.sum(np.isclose(utilities, best_value, atol=tolerance, rtol=0.0)))
        distinct = np.unique(np.sort(utilities))[::-1]
        margin = float(best_value - distinct[1]) if len(distinct) > 1 else 0.0
        detailed.setdefault(question, []).append((weight, best_count > 1, margin))

    def detailed_summary(items: Sequence[tuple[float, bool, float]]) -> Dict[str, Any]:
        total = sum(item[0] for item in items)
        return {
            "records": int(len(items)),
            "weighted_records": float(total),
            "best_action_tie_rate": float(sum(item[0] for item in items if item[1]) / max(1e-12, total)),
            "unique_best_action_rate": float(sum(item[0] for item in items if not item[1]) / max(1e-12, total)),
            "utility_best_second_margin": _summary([item[2] for item in items]),
        }

    output: Dict[str, Any] = {}
    for question, items in sorted(detailed.items()):
        output[question] = {}
        subsets = {
            "ALL": list(items),
            "UNIQUE_BEST": [item for item in items if not item[1]],
            "POSITIVE_MARGIN": [item for item in items if item[2] > tolerance],
        }
        for subset_name, subset in subsets.items():
            output[question][subset_name] = detailed_summary(subset)
    all_items = [item for items in detailed.values() for item in items]
    output["ALL_QUESTIONS"] = {}
    for subset_name, subset in {
        "ALL": all_items,
        "UNIQUE_BEST": [item for item in all_items if not item[1]],
        "POSITIVE_MARGIN": [item for item in all_items if item[2] > tolerance],
    }.items():
        output["ALL_QUESTIONS"][subset_name] = detailed_summary(subset)
    return {
        "tie_tolerance": tolerance,
        "by_question": output,
    }


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
    _assert_probability_contract(probabilities, legal)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise AssertionError("sample_weight must be finite and non-negative")
    denominator = float(weights.sum())
    if denominator <= 0:
        raise AssertionError("validation has no positive sample weight")
    nll = brier = correct = 0.0
    confidence = []
    action_distribution: Dict[str, float] = {}
    best_distribution: Dict[str, float] = {}
    mechanism: Dict[str, float] = {}
    for row in range(len(indices)):
        valid = legal[row] >= 0
        probs = probabilities[row]
        chosen_col = int(np.argmax(probs))
        if not valid[chosen_col]:
            raise AssertionError("policy selected a masked illegal action")
        chosen_id = int(legal[row, chosen_col])
        weight = float(weights[row])
        chosen_name = data.manifest["action_names"][chosen_id]
        hit = float(best[row, chosen_id])
        if hit not in (0.0, 1.0):
            raise AssertionError(f"correct must be binary, got {hit}")
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
    ece = _ece(confidence, denominator)
    oracle = 0.0
    oracle_weight = 0.0
    utility_index = OUTCOME_FIELDS.index("utility")
    for row in range(len(indices)):
        valid = legal[row] >= 0
        legal_ids = legal[row][valid].astype(np.int64)
        utility_values = outcomes[row, legal_ids, utility_index]
        finite_utility = utility_values[np.isfinite(utility_values)]
        if len(finite_utility) == 0:
            continue
        oracle += weights[row] * float(np.max(finite_utility))
        oracle_weight += weights[row]
    oracle_best_utility = None if oracle_weight <= 0 else oracle / oracle_weight
    return {
        "records": int(len(indices)),
        "weighted_records": denominator,
        "val_utility": mechanism.get("utility", 0.0) / denominator,
        "oracle_best_utility": oracle_best_utility,
        "nll": nll / denominator,
        "accuracy": correct / denominator,
        "brier": brier / denominator,
        "ece": ece,
        "action_distribution": {name: value / denominator for name, value in sorted(action_distribution.items())},
        "best_action_distribution": {name: value / denominator for name, value in sorted(best_distribution.items())},
        "mechanism": {field: value / denominator for field, value in sorted(mechanism.items())},
    }


def action_baseline_metrics(
    data: CompactJEVData,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
) -> Dict[str, Any]:
    """Report non-learned references on held-out sequences.

    The majority policy is fitted only from policy-training sequences.  The
    uniform policy is reported as an expectation over legal actions, so it
    does not depend on a process-global random stream.
    """
    counts: Dict[int, Dict[int, float]] = {}
    for index in np.asarray(train_indices, dtype=np.int64):
        if float(data.sample_weight[index]) <= 0:
            continue
        question = int(data.questions[index])
        row = counts.setdefault(question, {})
        legal = np.asarray(data.legal_actions[index])
        best = np.asarray(data.best_mask[index])
        for action_id in legal[legal >= 0]:
            if bool(best[int(action_id)]):
                row[int(action_id)] = row.get(int(action_id), 0.0) + float(data.sample_weight[index])

    methods: Dict[str, Dict[str, Any]] = {}
    utility_index = OUTCOME_FIELDS.index("utility")
    for mode in ("majority_action", "uniform_legal_action"):
        total_weight = 0.0
        correct = 0.0
        utility = 0.0
        action_distribution: Dict[str, float] = {}
        for index in np.asarray(val_indices, dtype=np.int64):
            weight = float(data.sample_weight[index])
            if weight <= 0:
                continue
            legal = [int(value) for value in np.asarray(data.legal_actions[index]) if int(value) >= 0]
            if not legal:
                continue
            if mode == "majority_action":
                question_counts = counts.get(int(data.questions[index]), {})
                chosen = max(legal, key=lambda action: (question_counts.get(action, 0.0), -action))
                choices = [chosen]
            else:
                choices = legal
            best = np.asarray(data.best_mask[index])
            outcomes = np.asarray(data.outcomes[index], dtype=np.float64)
            correct += weight * sum(float(best[action]) for action in choices) / len(choices)
            utility += weight * sum(float(outcomes[action, utility_index]) for action in choices) / len(choices)
            for action in choices:
                name = data.manifest["action_names"][action]
                action_distribution[name] = action_distribution.get(name, 0.0) + weight / len(choices)
            total_weight += weight
        methods[mode] = {
            "records": int(len(val_indices)),
            "weighted_records": total_weight,
            "best_action_accuracy": correct / max(total_weight, 1e-8),
            "val_utility": utility / max(total_weight, 1e-8),
            "action_distribution": {
                name: value / max(total_weight, 1e-8)
                for name, value in sorted(action_distribution.items())
            },
            "fit_scope": "policy_train_sequences" if mode == "majority_action" else "none",
        }
    return methods


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_first_round_markdown(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# H=8 JEV Three-Way First-Round Validation",
        "",
        "This is the speed-only first round: one fixed seed (`20261003`) per method on the same full TRAIN H=8 dataset.",
        "Multi-seed mean/std, ensemble, and robustness metrics are intentionally deferred.",
        "Official TEST data is not read for this protocol.",
        "",
        "## Locked protocol",
        "",
        f"- Seed: `{report['seeds'][0]}`; sequence-disjoint split is fixed before training.",
        f"- Training: `{report['training']['epochs']}` epochs, batch `{report['training']['batch_size']}`, AdamW, LR `{report['training']['learning_rate']}`.",
        "- All methods consume identical state vectors, questions, legal actions, target probabilities, and sample weights.",
        "- Temperature scaling is fitted on policy validation only and applied to all three methods.",
        "",
        "## First-round validation metrics",
        "",
        "| Method | Params | Val NLL | Best-action Accuracy | Brier | ECE | Validation Utility |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["methods"].values():
        metrics = item["validation"]
        lines.append(
            f"| {item['label']} | {item['trainable_params']:,} | "
            f"{metrics['nll']:.6f} | {metrics['accuracy']:.6f} | "
            f"{metrics['brier']:.6f} | {metrics['ece']:.6f} | "
            f"{metrics['val_utility']:.6f} |"
        )
    lines += [
        "",
        "## Gate for tracking comparison",
        "",
        f"`jev_beats_both_on_val_utility = {report['jev_beats_both_on_val_utility']}`.",
        f"`jev_beats_majority_reference = {report['jev_beats_majority_reference']}`; `jev_matches_majority_reference = {report['jev_matches_majority_reference']}`.",
        "A majority tie is not evidence of a meaningful learned-policy advantage; runtime parity and reactivation coverage are separate hard gates.",
        "",
        "## Non-learned validation references",
        "",
        "| Reference | Best-action Accuracy | Validation Utility |",
        "|---|---:|---:|",
    ]
    for name, baseline in report["baselines"].items():
        lines.append(
            f"| {name} | {baseline['best_action_accuracy']:.6f} | "
            f"{baseline['val_utility']:.6f} |"
        )
    lines += [
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
    data = CompactJEVData(args.dataset.resolve())
    split = load_policy_split(args.split_manifest.resolve())
    train_indices = data.indices_for_sequences(split["train"])
    train_indices = train_indices[np.asarray(data.sample_weight[train_indices]) > 0]
    val_indices = data.indices_for_sequences(split["val"])
    val_indices = val_indices[np.asarray(data.sample_weight[val_indices]) > 0]
    if len(val_indices) == 0:
        raise ValueError("validation split has no positive-weight compact records")

    report_methods: Dict[str, Any] = {}
    training_configs = []
    for method, label in METHODS.items():
        root = args.methods_root.resolve() / method
        manifest = json.loads((root / "method_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "PASS" or list(manifest.get("seeds", ())) != list(SEEDS):
            raise ValueError(f"method manifest incomplete: {root}")
        training_configs.append(dict(manifest["training"]))
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
        if len(item["seed_metrics"]) != 1 or item["seed_metrics"][0]["seed"] != 20261003:
            raise ValueError("first-round report requires exactly seed 20261003")
        # Keep the one observed validation result directly.  No mean/std or
        # robustness statistic is computed in this temporary speed-only round.
        item["validation"] = dict(item["seed_metrics"][0])
        item["relative_param_difference"] = abs(item["trainable_params"] - target_params) / max(1, target_params)
        if item["relative_param_difference"] >= 0.02:
            raise ValueError(f"parameter matching gate failed for {item['label']}")
    jev_utility = report_methods["jev"]["validation"]["val_utility"]
    shared_training_fields = ("epochs", "batch_size", "optimizer", "learning_rate")
    if any(
        any(config.get(field) != training_configs[0].get(field) for field in shared_training_fields)
        for config in training_configs[1:]
    ):
        raise ValueError("three-way methods do not share the locked training configuration")
    baselines = action_baseline_metrics(data, train_indices, val_indices)
    oracle_best_utility = report_methods["jev"]["validation"]["oracle_best_utility"]
    if oracle_best_utility is None or not math.isfinite(float(oracle_best_utility)):
        raise AssertionError("oracle_best_utility is not finite")
    for method, item in report_methods.items():
        observed = float(item["validation"]["val_utility"])
        if observed > float(oracle_best_utility) + 1e-6:
            raise AssertionError(
                f"{method} val_utility exceeds oracle_best_utility: "
                f"{observed} > {oracle_best_utility}"
            )
    for baseline, item in baselines.items():
        item["oracle_best_utility"] = float(oracle_best_utility)
        observed = float(item["val_utility"])
        if observed > float(oracle_best_utility) + 1e-6:
            raise AssertionError(
                f"{baseline} val_utility exceeds oracle_best_utility: "
                f"{observed} > {oracle_best_utility}"
            )
    tie_margin_diagnostics = utility_tie_margin_diagnostics(data, val_indices)
    majority_utility = float(baselines["majority_action"]["val_utility"])
    jev_beats_majority = bool(jev_utility > majority_utility + 1e-6)
    jev_matches_majority = bool(abs(jev_utility - majority_utility) <= 1e-6)
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
        "seed_policy": "TEMPORARILY_DISABLED_MULTI_SEED",
        "multi_seed_robustness": "deferred_until_after_first_round_gate",
        "training": {
            **{field: training_configs[0][field] for field in shared_training_fields},
            "calibration": "temperature_scaling_policy_val_only",
            "hidden_dim_by_method": {
                method: int(report_methods[method]["hidden_dim"]) for method in report_methods
            },
        },
        "methods": report_methods,
        "baselines": baselines,
        "oracle_invariant": {
            "oracle_best_utility": float(oracle_best_utility),
            "learned_policies_le_oracle": True,
            "majority_le_oracle": True,
            "uniform_le_oracle": True,
            "tolerance": 1e-6,
        },
        "tie_margin_diagnostics": tie_margin_diagnostics,
        "jev_beats_both_on_val_utility": bool(
            jev_utility > report_methods["question_threshold"]["validation"]["val_utility"]
            and jev_utility > report_methods["question_conditioned_mlp"]["validation"]["val_utility"]
        ),
        "jev_beats_majority_reference": jev_beats_majority,
        "jev_matches_majority_reference": jev_matches_majority,
        "meaningful_learned_policy_advantage": bool(
            jev_utility > report_methods["question_threshold"]["validation"]["val_utility"]
            and jev_utility > report_methods["question_conditioned_mlp"]["validation"]["val_utility"]
            and jev_beats_majority
        ),
        "official_test_read": False,
        "tracking_gate": "runtime parity and reactivation coverage are required; offline JEV must beat both learned controls and the majority reference for a meaningful architecture claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_first_round_markdown(report, args.markdown)
    print(json.dumps({"status": "PASS", "output": str(args.output), "jev_beats_both_on_val_utility": report["jev_beats_both_on_val_utility"]}, indent=2))


if __name__ == "__main__":
    main()
