"""Fit a scalar probability temperature on policy validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch

from gtr.modeling.jev_decision import ACTION_NAMES
from gtr.modeling.jev_runtime import build_controller_from_checkpoint
from jev_metrics import summarize
from train_jev import load_policy_split, read_records


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


@torch.no_grad()
def predictions(model, records: Sequence[Mapping[str, Any]], temperature: float = 1.0):
    result = []
    for record in records:
        features = torch.tensor(
            [record["state"]["feature_vector"]], dtype=torch.float32
        )
        output = model(
            features,
            [record["question_type"]],
            [record["legal_actions"]],
        )
        values = output["probs"][0, : len(record["legal_actions"])].clamp_min(1e-8)
        if temperature != 1.0:
            values = torch.softmax(values.log() / float(temperature), dim=0)
        result.append(
            {
                "legal_actions": list(record["legal_actions"]),
                "probabilities": [float(value) for value in values.tolist()],
            }
        )
    return result


def weighted_records(records):
    return [record for record in records if float(record.get("sample_weight", 1.0)) > 0]


def fit_temperature(model, records: Sequence[Mapping[str, Any]]) -> float:
    records = weighted_records(records)
    if not records:
        return 1.0
    raw = predictions(model, records)
    log_probs = [
        torch.log(torch.tensor(prediction["probabilities"], dtype=torch.float64))
        for prediction in raw
    ]
    targets = [
        torch.tensor(
            [
                dict(zip(record["legal_actions"], record["target_probs"]))[name]
                for name in prediction["legal_actions"]
            ],
            dtype=torch.float64,
        )
        for record, prediction in zip(records, raw)
    ]
    weights = [float(record.get("sample_weight", 1.0)) for record in records]
    parameter = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    optimizer = torch.optim.LBFGS([parameter], lr=0.25, max_iter=80, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = parameter.exp().clamp(0.05, 20.0)
        total = torch.tensor(0.0, dtype=torch.float64)
        denominator = sum(weights)
        for log_prob, target, weight in zip(log_probs, targets, weights):
            scaled = torch.log_softmax(log_prob / temperature, dim=0)
            total = total - float(weight) * (target * scaled).sum()
        loss = total / max(denominator, 1e-8)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(parameter.detach().exp().clamp(0.05, 20.0).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    split = load_policy_split(args.policy_split)
    records = read_records(args.dataset)
    val_names = set(split["val"])
    val = [record for record in records if record["sequence"] in val_names]
    if not val:
        raise ValueError("policy validation split contains no records")
    model = build_controller_from_checkpoint(args.checkpoint, device=args.device)
    useful = weighted_records(val)
    before = predictions(model, useful)
    temperature = fit_temperature(model, val)
    after = predictions(model, useful, temperature=temperature)
    report = {
        "status": "PASS",
        "calibration_fit": "policy_val_only",
        "dataset": str(args.dataset.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "policy_split": str(args.policy_split.resolve()),
        "policy_split_sha256": sha256(args.policy_split),
        "validation_records": len(val),
        "weighted_validation_records": len(useful),
        "temperature": temperature,
        "before": summarize(useful, before),
        "after": summarize(useful, after),
        "official_test_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

