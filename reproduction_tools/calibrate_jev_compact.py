"""Fit validation-only temperature scaling on the shared compact dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from gtr.modeling.jev_runtime import build_controller_from_checkpoint
from jev_compact_dataset import CompactJEVData
from train_jev import load_policy_split


def predict(model, data: CompactJEVData, indices: np.ndarray):
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), 2048):
            group = indices[start : start + 2048]
            features = torch.as_tensor(np.asarray(data.features[group]), dtype=torch.float32)
            questions = torch.as_tensor(np.asarray(data.questions[group]), dtype=torch.long)
            legal = torch.as_tensor(np.asarray(data.legal_actions[group]), dtype=torch.long)
            output = model(features, questions, legal)
            rows.append(output["probs"].cpu())
    return torch.cat(rows, dim=0) if rows else torch.empty((0, len(data.manifest["action_names"])))


def metrics(probabilities: torch.Tensor, data: CompactJEVData, indices: np.ndarray, temperature: float = 1.0):
    legal = torch.as_tensor(np.asarray(data.legal_actions[indices]), dtype=torch.long)
    target = torch.as_tensor(np.asarray(data.target_probs[indices]), dtype=torch.float64)
    weights = torch.as_tensor(np.asarray(data.sample_weight[indices]), dtype=torch.float64)
    valid = legal >= 0
    logits = probabilities.double().clamp_min(1e-8).log()
    if temperature != 1.0:
        logits = logits / float(temperature)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        probabilities = torch.softmax(logits.float(), dim=-1)
    probs = probabilities.double()
    denom = weights.sum().clamp_min(1e-8)
    nll = (-(target * probs.clamp_min(1e-8).log()).sum(dim=1) * weights).sum() / denom
    brier = (((probs - target) ** 2).masked_fill(~valid, 0.0).sum(dim=1) * weights).sum() / denom
    chosen = probs.argmax(dim=1)
    best = torch.as_tensor(np.asarray(data.best_mask[indices]), dtype=torch.bool)
    chosen_ids = legal.gather(1, chosen[:, None]).squeeze(1)
    correct = best.gather(1, chosen_ids.clamp_min(0)[:, None]).squeeze(1).double()
    confidence = probs.gather(1, chosen[:, None]).squeeze(1)
    accuracy = (correct * weights).sum() / denom
    ece = torch.tensor(0.0, dtype=torch.float64)
    for bucket in range(10):
        lower, upper = bucket / 10.0, (bucket + 1) / 10.0
        selected = (confidence >= lower) & ((confidence < upper) | ((bucket == 9) & (confidence <= upper)))
        bucket_weight = weights.masked_fill(~selected, 0.0).sum()
        if bucket_weight > 0:
            ece += bucket_weight / denom * abs(
                (confidence * weights).masked_fill(~selected, 0.0).sum() / bucket_weight
                - (correct * weights).masked_fill(~selected, 0.0).sum() / bucket_weight
            )
    return {
        "records": int(len(indices)),
        "weighted_records": float(denom),
        "nll": float(nll),
        "brier": float(brier),
        "ece": float(ece),
        "best_action_accuracy": float(accuracy),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = CompactJEVData(args.dataset.resolve())
    split = load_policy_split(args.policy_split.resolve())
    indices = data.indices_for_sequences(split["val"])
    indices = indices[np.asarray(data.sample_weight[indices]) > 0]
    if len(indices) == 0:
        raise ValueError("validation has no positive-weight compact records")
    model = build_controller_from_checkpoint(args.checkpoint.resolve(), device="cpu")
    raw = predict(model, data, indices)
    target = torch.as_tensor(np.asarray(data.target_probs[indices]), dtype=torch.float64)
    weights = torch.as_tensor(np.asarray(data.sample_weight[indices]), dtype=torch.float64)
    valid = torch.as_tensor(np.asarray(data.legal_actions[indices]), dtype=torch.long) >= 0
    log_probs = raw.double().clamp_min(1e-8).log()
    parameter = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    optimizer = torch.optim.LBFGS([parameter], lr=0.25, max_iter=80, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = parameter.exp().clamp(0.05, 20.0)
        scaled = (log_probs / temperature).masked_fill(~valid, torch.finfo(torch.float64).min)
        log_probability = torch.log_softmax(scaled, dim=1)
        loss = (-(target * log_probability).sum(dim=1) * weights).sum() / weights.sum().clamp_min(1e-8)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(parameter.detach().exp().clamp(0.05, 20.0))
    report = {
        "status": "PASS",
        "calibration_fit": "policy_val_only",
        "dataset": str(args.dataset.resolve()),
        "dataset_manifest_sha256": sha256(args.dataset.resolve() / "manifest.json"),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint.resolve()),
        "policy_split": str(args.policy_split.resolve()),
        "validation_records": int(len(indices)),
        "weighted_validation_records": float(weights.sum()),
        "temperature": temperature,
        "before": metrics(raw, data, indices),
        "after": metrics(raw, data, indices, temperature),
        "official_test_read": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "calibration_val_only.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    payload = torch.load(args.checkpoint.resolve(), map_location="cpu")
    payload["temperature"] = temperature
    torch.save(payload, args.output / "model_calibrated.pth")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
