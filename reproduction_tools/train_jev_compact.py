"""Train one JEV policy directly from the shared memory-mapped H=8 arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch

from gtr.modeling.jev_decision import QUESTION_NAMES
from jev_compact_dataset import CompactJEVData
from train_jev import choose_model, load_policy_split


def assert_probability_contract(
    probabilities: torch.Tensor,
    legal: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    *,
    tolerance: float = 1e-6,
) -> None:
    """Fail closed on invalid masked distributions and weights."""

    if not torch.isfinite(probabilities).all():
        raise AssertionError("policy probabilities contain non-finite values")
    if bool((probabilities < -tolerance).any()) or bool((probabilities > 1.0 + tolerance).any()):
        raise AssertionError("policy probabilities are outside [0, 1]")
    valid = legal >= 0
    if not bool(valid.any(dim=1).all()):
        raise AssertionError("a batch record has no legal action")
    legal_sum = probabilities.masked_fill(~valid, 0.0).sum(dim=1)
    if bool((legal_sum - 1.0).abs().gt(tolerance).any()):
        raise AssertionError("legal-action probabilities do not sum to one")
    if bool(probabilities.masked_fill(valid, 0.0).abs().gt(tolerance).any()):
        raise AssertionError("masked illegal-action probabilities are non-zero")
    if sample_weight is not None:
        if not torch.isfinite(sample_weight).all() or bool((sample_weight < 0).any()):
            raise AssertionError("sample_weight must be finite and non-negative")


def batch_forward(model, data: CompactJEVData, indices: np.ndarray, device: torch.device):
    features = torch.as_tensor(np.asarray(data.features[indices]), dtype=torch.float32, device=device)
    questions = torch.as_tensor(np.asarray(data.questions[indices]), dtype=torch.long, device=device)
    legal = torch.as_tensor(np.asarray(data.legal_actions[indices]), dtype=torch.long, device=device)
    targets = torch.as_tensor(np.asarray(data.target_probs[indices]), dtype=torch.float32, device=device)
    weights = torch.as_tensor(np.asarray(data.sample_weight[indices]), dtype=torch.float32, device=device)
    output = model(features, questions, legal)
    assert_probability_contract(output["probs"], legal, weights)
    if not torch.isfinite(targets).all():
        raise AssertionError("target probabilities contain non-finite values")
    loss_rows = -(targets * output["probs"].clamp_min(1e-8).log()).sum(dim=1)
    return (loss_rows * weights).sum() / weights.sum().clamp_min(1.0), output, loss_rows, weights


@torch.no_grad()
def evaluate(model, data: CompactJEVData, indices: np.ndarray, device: torch.device) -> Dict[str, Any]:
    model.eval()
    if len(indices) == 0:
        return {"records": 0, "weighted_records": 0.0}
    total_weight = 0.0
    nll = brier = correct = 0.0
    confidence = []
    ignored = 0
    by_question: Dict[int, list[float]] = {}
    for start in range(0, len(indices), 2048):
        group = indices[start : start + 2048]
        _loss, output, loss_rows, weights = batch_forward(model, data, group, device)
        probs = output["probs"].detach().cpu().numpy()
        legal = np.asarray(data.legal_actions[group])
        targets = np.asarray(data.target_probs[group])
        best = np.asarray(data.best_mask[group])
        questions = np.asarray(data.questions[group])
        row_weights = np.asarray(data.sample_weight[group], dtype=np.float64)
        chosen_columns = probs.argmax(axis=1)
        for row in range(len(group)):
            weight = float(row_weights[row])
            if not np.isfinite(weight) or weight < 0:
                raise AssertionError("sample_weight must be finite and non-negative")
            if weight <= 0:
                ignored += 1
                continue
            chosen_id = int(legal[row, chosen_columns[row]])
            hit = float(best[row, chosen_id]) if 0 <= chosen_id < best.shape[1] else 0.0
            if hit not in (0.0, 1.0):
                raise AssertionError(f"correct must be binary, got {hit}")
            valid = legal[row] >= 0
            nll += weight * float(-(targets[row, valid] * np.log(np.maximum(probs[row, valid], 1e-8))).sum())
            brier += weight * float(((probs[row, valid] - targets[row, valid]) ** 2).sum())
            correct += weight * hit
            total_weight += weight
            confidence.append((float(probs[row, chosen_columns[row]]), hit, weight))
            by_question.setdefault(int(questions[row]), []).append(hit)
    ece = 0.0
    for bucket in range(10):
        lower, upper = bucket / 10.0, (bucket + 1) / 10.0
        selected = [
            item
            for item in confidence
            if lower <= item[0] < upper
            or (bucket == 9 and lower <= item[0] <= upper)
        ]
        bucket_weight = sum(item[2] for item in selected)
        if bucket_weight:
            ece += bucket_weight / total_weight * abs(
                sum(item[0] * item[2] for item in selected) / bucket_weight
                - sum(item[1] * item[2] for item in selected) / bucket_weight
            )
    if not np.isfinite(ece) or not (0.0 <= ece <= 1.0 + 1e-12):
        raise AssertionError(f"ECE is outside [0, 1]: {ece}")
    accuracy = correct / max(total_weight, 1e-8)
    if not (0.0 <= accuracy <= 1.0 + 1e-12):
        raise AssertionError(f"accuracy is outside [0, 1]: {accuracy}")
    return {
        "records": int(len(indices)),
        "nll": nll / max(total_weight, 1e-8),
        "brier": brier / max(total_weight, 1e-8),
        "ece": min(1.0, max(0.0, ece)),
        "best_action_accuracy": accuracy,
        "weighted_records": total_weight,
        "ignored_uninformative_records": ignored,
        "by_question": {
            QUESTION_NAMES[question] if question < len(QUESTION_NAMES) else str(question): {
                "records": len(values),
                "accuracy": sum(values) / len(values),
            }
            for question, values in sorted(by_question.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.hidden_dim < 1 or args.lr <= 0:
        raise ValueError("training arguments must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = CompactJEVData(args.dataset.resolve())
    splits = load_policy_split(args.split_manifest.resolve())
    train_indices = data.indices_for_sequences(splits["train"])
    val_indices = data.indices_for_sequences(splits["val"])
    if len(train_indices) == 0 or len(val_indices) == 0:
        raise ValueError("policy split has an empty compact subset")
    state_dim = int(data.manifest["state_dim"])
    device = torch.device(args.device)
    model = choose_model(args.model, state_dim, args.hidden_dim).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr) if parameters else None
    rng = np.random.default_rng(args.seed)
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = train_indices.copy()
        rng.shuffle(order)
        total_loss = 0.0
        total_weight = 0.0
        for start in range(0, len(order), args.batch_size):
            group = order[start : start + args.batch_size]
            loss, _output, _rows, weights = batch_forward(model, data, group, device)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            batch_weight = float(np.asarray(data.sample_weight[group], dtype=np.float64).sum())
            total_loss += float(loss.item()) * batch_weight
            total_weight += batch_weight
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / max(total_weight, 1e-8),
                "val": evaluate(model, data, val_indices, device),
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_name": args.model,
            "threshold": None,
            "state_dim": state_dim,
            "hidden_dim": args.hidden_dim,
            "question_dim": max(8, args.hidden_dim // 4)
            if args.model in {"jev", "question_threshold", "question_conditioned_mlp"}
            else None,
            "action_dim": max(8, args.hidden_dim // 4) if args.model == "jev" else None,
            "num_layers": 2 if args.model == "jev" else None,
            "temperature": 1.0 if args.model == "jev" else None,
            "use_option_interaction": False if args.model == "jev" else None,
            "seed": args.seed,
            "splits": splits,
            "compact_dataset": str(args.dataset.resolve()),
            "compact_dataset_manifest_sha256": __import__("hashlib").sha256(
                (args.dataset.resolve() / "manifest.json").read_bytes()
            ).hexdigest(),
        },
        args.output / "model.pth",
    )
    report = {
        "status": "COMPLETE",
        "model": args.model,
        "dataset": str(args.dataset.resolve()),
        "state_dim": state_dim,
        "seed": args.seed,
        "splits": {"train": int(len(train_indices)), "val": int(len(val_indices)), "test": 0},
        "history": history,
        "test": {"records": 0},
        "future_gt_used_by_model": False,
        "compact_dataset": True,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
