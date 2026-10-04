"""Train/evaluate a JEV policy or an interface-compatible baseline.

This is an offline decision trainer.  It validates future-GT records before
reading only their current-state ``feature_vector``; no future outcome is
passed to a model.  The generated checkpoint is suitable for SHADOW/JEV
runtime use only after the OFF equivalence gate has passed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from gtr.modeling.jev_baselines import (
    FixedThresholdPolicy,
    FixedSlotMLP,
    GlobalLearnedThreshold,
    IndependentMLPHeads,
    LogisticGate,
    SharedEncoderSeparateHeads,
    StateConditionedThreshold,
)
from gtr.modeling.jev_decision import JEVDecisionController
from gtr.modeling.jev_decision import ACTION_NAMES
from jev_metrics import summarize as summarize_decision_metrics
from jev_dataset_contract import validate_record
from jev_dataset_tools import split_sequences


MODEL_NAMES = {
    "fixed_threshold",
    "jev",
    "fixed_slot_mlp",
    "independent_mlp",
    "shared_heads",
    "logistic",
    "global_threshold",
    "state_threshold",
}


def read_records(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                validate_record(record, allow_future_gt=True)
            except Exception as exc:
                raise ValueError(f"invalid record at line {line_number}: {exc}") from exc
            features = record["state"].get("feature_vector")
            if not isinstance(features, list) or not features:
                raise ValueError("each state must contain a numeric feature_vector")
            if not all(isinstance(value, (int, float)) for value in features):
                raise ValueError("feature_vector must contain only numeric values")
            if not np.isfinite(np.asarray(features, dtype=np.float32)).all():
                raise ValueError("feature_vector contains a non-finite value")
            records.append(record)
    if not records:
        raise ValueError("empty JEV dataset")
    dimensions = {len(record["state"]["feature_vector"]) for record in records}
    if len(dimensions) != 1:
        raise ValueError(f"feature_vector dimension mismatch: {sorted(dimensions)}")
    return records


def choose_model(name: str, state_dim: int, hidden_dim: int, fixed_threshold: float = 0.2) -> nn.Module:
    if name == "fixed_threshold":
        return FixedThresholdPolicy(threshold=fixed_threshold)
    if name == "jev":
        return JEVDecisionController(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            question_dim=max(8, hidden_dim // 4),
            action_dim=max(8, hidden_dim // 4),
        )
    if name == "fixed_slot_mlp":
        return FixedSlotMLP(state_dim, hidden_dim)
    if name == "independent_mlp":
        return IndependentMLPHeads(state_dim, hidden_dim)
    if name == "shared_heads":
        return SharedEncoderSeparateHeads(state_dim, hidden_dim)
    if name == "logistic":
        return LogisticGate(state_dim)
    if name == "global_threshold":
        return GlobalLearnedThreshold()
    if name == "state_threshold":
        if state_dim < 4:
            raise ValueError("state_threshold requires the four-column baseline prefix")
        return StateConditionedThreshold(state_dim)
    raise ValueError(f"unknown model: {name}")


def batches(records: Sequence[Dict[str, Any]], batch_size: int, rng: random.Random):
    order = list(range(len(records)))
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [records[index] for index in order[start : start + batch_size]]


def batch_tensors(records: Sequence[Dict[str, Any]], device: torch.device):
    features = torch.tensor(
        [record["state"]["feature_vector"] for record in records],
        dtype=torch.float32,
        device=device,
    )
    questions = [record["question_type"] for record in records]
    legal = [record["legal_actions"] for record in records]
    targets = [record["target_probs"] for record in records]
    return features, questions, legal, targets


def forward_loss(model: nn.Module, records: Sequence[Dict[str, Any]], device: torch.device):
    features, questions, legal, targets = batch_tensors(records, device)
    output = model(features, questions, legal)
    losses = []
    for row, target in enumerate(targets):
        target_tensor = torch.tensor(target, dtype=torch.float32, device=device)
        probability = output["probs"][row, : len(target)].clamp_min(1e-8)
        losses.append(-(target_tensor * probability.log()).sum())
    return torch.stack(losses).mean(), output


@torch.no_grad()
def evaluate(model: nn.Module, records: Sequence[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
    model.eval()
    if not records:
        return {"records": 0}
    correct = 0
    nll = 0.0
    by_question: Dict[str, List[float]] = {}
    metric_records: List[Mapping[str, Any]] = []
    metric_predictions: List[Mapping[str, Any]] = []
    for start in range(0, len(records), 256):
        group = records[start : start + 256]
        loss, output = forward_loss(model, group, device)
        nll += float(loss.item()) * len(group)
        predicted = output["probs"].argmax(dim=1)
        for row, record in enumerate(group):
            chosen = output["legal_actions"][row, predicted[row]].item()
            chosen_name = (
                ACTION_NAMES[chosen]
            )
            hit = float(chosen_name in record["best_actions"])
            correct += int(hit)
            by_question.setdefault(record["question_type"], []).append(hit)
            metric_records.append(record)
            metric_predictions.append(
                {
                    "legal_actions": [
                        ACTION_NAMES[index]
                        for index in output["legal_actions"][row, : len(record["legal_actions"])].detach().cpu().tolist()
                    ],
                    "probabilities": [
                        float(value)
                        for value in output["probs"][row, : len(record["legal_actions"])].detach().cpu().tolist()
                    ],
                }
            )
    decision_metrics = summarize_decision_metrics(metric_records, metric_predictions)
    return {
        **decision_metrics,
        "nll": nll / len(records),
        "best_action_accuracy": correct / len(records),
        "by_question": {
            key: {"records": len(values), "accuracy": sum(values) / len(values)}
            for key, values in sorted(by_question.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(MODEL_NAMES), default="jev")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fixed-threshold", type=float, default=0.2)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError("epochs, batch-size and lr must be positive")
    if args.model not in MODEL_NAMES:
        raise ValueError(args.model)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = read_records(args.dataset)
    splits = split_sequences(
        [record["sequence"] for record in records], seed=args.seed
    )
    split_sets = {key: set(value) for key, value in splits.items()}
    grouped = {
        key: [record for record in records if record["sequence"] in split_sets[key]]
        for key in splits
    }
    state_dim = len(records[0]["state"]["feature_vector"])
    device = torch.device(args.device)
    model = choose_model(args.model, state_dim, args.hidden_dim, args.fixed_threshold).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr) if parameters else None
    rng = random.Random(args.seed)
    history = []
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        count = 0
        for group in batches(grouped["train"], args.batch_size, rng):
            loss, _ = forward_loss(model, group, device)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            epoch_loss += float(loss.item()) * len(group)
            count += len(group)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": epoch_loss / max(1, count),
                "val": evaluate(model, grouped["val"], device),
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_name": args.model,
            "threshold": args.fixed_threshold if args.model == "fixed_threshold" else None,
            "state_dim": state_dim,
            "hidden_dim": args.hidden_dim,
            "question_dim": max(8, args.hidden_dim // 4) if args.model == "jev" else None,
            "action_dim": max(8, args.hidden_dim // 4) if args.model == "jev" else None,
            "num_layers": 2 if args.model == "jev" else None,
            "temperature": 1.0 if args.model == "jev" else None,
            "use_option_interaction": False if args.model == "jev" else None,
            "seed": args.seed,
            "splits": splits,
        },
        args.output / "model.pth",
    )
    report = {
        "status": "COMPLETE",
        "model": args.model,
        "dataset": str(args.dataset),
        "state_dim": state_dim,
        "seed": args.seed,
        "splits": {key: len(value) for key, value in grouped.items()},
        "history": history,
        "test": evaluate(model, grouped["test"], device),
        "future_gt_used_by_model": False,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
