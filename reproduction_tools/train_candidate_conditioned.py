"""Train and evaluate ID-free candidate-conditioned MATCH scorers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from jev_candidate_model import CandidateConditionedScorer


SEED = 20261003


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema_version") != "jev_candidate_match_v1":
                raise ValueError(f"{path}:{line_number}: unsupported candidate schema")
            candidates = list(record.get("candidates", ()))
            keys = [str(row.get("candidate_key")) for row in candidates]
            if len(keys) != len(set(keys)) or "START_NEW" not in keys:
                raise ValueError(f"{path}:{line_number}: invalid candidate keys")
            if set(record.get("best_candidates", ())) - set(keys):
                raise ValueError(f"{path}:{line_number}: best candidate outside legal set")
            target = record.get("target_probs", {})
            if set(target) != set(keys):
                raise ValueError(f"{path}:{line_number}: target keys mismatch")
            if abs(sum(float(value) for value in target.values()) - 1.0) > 1e-5:
                raise ValueError(f"{path}:{line_number}: target does not sum to one")
            if not record.get("uses_future_gt"):
                raise ValueError(f"{path}:{line_number}: future-GT provenance missing")
            for row in candidates:
                if len(row.get("features", ())) != 8:
                    raise ValueError(f"{path}:{line_number}: candidate feature dimension mismatch")
                if not np.isfinite(np.asarray(row["features"], dtype=np.float32)).all():
                    raise ValueError(f"{path}:{line_number}: non-finite candidate feature")
                if not math.isfinite(float(row["outcome"]["utility"])):
                    raise ValueError(f"{path}:{line_number}: non-finite candidate utility")
            records.append(record)
    if not records:
        raise ValueError(f"empty candidate dataset: {path}")
    return records


def provenance(records: Sequence[Mapping[str, Any]], path: Path) -> Dict[str, Any]:
    values = {
        (
            str(record["provenance"]["source_commit"]),
            str(record["provenance"]["gmt_checkpoint_sha256"]),
            str(record["provenance"]["annotations_sha256"]),
            int(record["horizon"]),
        )
        for record in records
    }
    if len(values) != 1:
        raise ValueError(f"mixed candidate provenance in {path}: {sorted(values)!r}")
    source_commit, checkpoint_hash, annotations_hash, horizon = next(iter(values))
    return {
        "source_commit": source_commit,
        "gmt_checkpoint_sha256": checkpoint_hash,
        "annotations_sha256": annotations_hash,
        "horizon": horizon,
        "dataset_sha256": sha256(path),
    }


def collate(records: Sequence[Mapping[str, Any]], device: torch.device) -> Dict[str, Any]:
    if not records:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(records)
    state_dim = len(records[0]["state_features"])
    width = max(len(record["candidates"]) for record in records)
    state = torch.zeros((batch_size, state_dim), dtype=torch.float32, device=device)
    features = torch.zeros((batch_size, width, 8), dtype=torch.float32, device=device)
    mask = torch.zeros((batch_size, width), dtype=torch.bool, device=device)
    target = torch.zeros((batch_size, width), dtype=torch.float32, device=device)
    utility = torch.zeros((batch_size, width), dtype=torch.float32, device=device)
    keys: List[List[str]] = []
    best: List[set[str]] = []
    native: List[str | None] = []
    weights = torch.zeros(batch_size, dtype=torch.float32, device=device)
    for row_index, record in enumerate(records):
        state[row_index] = torch.tensor(record["state_features"], dtype=torch.float32, device=device)
        rows = list(record["candidates"])
        row_keys = []
        for column, candidate in enumerate(rows):
            row_keys.append(str(candidate["candidate_key"]))
            features[row_index, column] = torch.tensor(candidate["features"], dtype=torch.float32, device=device)
            target[row_index, column] = float(record["target_probs"][row_keys[-1]])
            utility[row_index, column] = float(candidate["outcome"]["utility"])
            mask[row_index, column] = True
        keys.append(row_keys)
        best.append({str(value) for value in record["best_candidates"]})
        native.append(record.get("native_candidate_key"))
        weights[row_index] = float(record.get("sample_weight", 1.0))
    return {
        "state": state,
        "features": features,
        "mask": mask,
        "target": target,
        "utility": utility,
        "keys": keys,
        "best": best,
        "native": native,
        "weights": weights,
    }


def forward_batch(model: CandidateConditionedScorer, records: Sequence[Mapping[str, Any]], device: torch.device):
    batch = collate(records, device)
    output = model(batch["state"], batch["features"], batch["mask"])
    losses = -(batch["target"] * output["probs"].clamp_min(1e-8).log()).masked_fill(~batch["mask"], 0.0).sum(dim=1)
    denominator = batch["weights"].sum().clamp_min(1.0)
    loss = (losses * batch["weights"]).sum() / denominator
    return loss, output, batch, losses


def _ece(confidence: List[Tuple[float, float, float]], total_weight: float) -> float:
    result = 0.0
    for bucket in range(10):
        lower, upper = bucket / 10.0, (bucket + 1) / 10.0
        rows = [
            item for item in confidence
            if lower <= item[0] < upper or (bucket == 9 and lower <= item[0] <= upper)
        ]
        weight = sum(item[2] for item in rows)
        if weight:
            mean_conf = sum(item[0] * item[2] for item in rows) / weight
            mean_hit = sum(item[1] * item[2] for item in rows) / weight
            result += weight / max(total_weight, 1e-8) * abs(mean_conf - mean_hit)
    return min(1.0, max(0.0, result))


@torch.no_grad()
def evaluate(model: CandidateConditionedScorer, records: Sequence[Mapping[str, Any]], device: torch.device) -> Dict[str, Any]:
    model.eval()
    if not records:
        return {"records": 0}
    nll = brier = correct = expected_utility = native_utility = random_utility = oracle_utility = 0.0
    total_weight = 0.0
    confidence: List[Tuple[float, float, float]] = []
    predicted_counts: Dict[str, int] = {}
    native_missing = 0
    for start in range(0, len(records), 128):
        group = records[start : start + 128]
        _loss, output, batch, losses = forward_batch(model, group, device)
        probabilities = output["probs"].detach().cpu().numpy()
        for row, record in enumerate(group):
            weight = float(batch["weights"][row].item())
            if weight <= 0:
                continue
            valid = batch["mask"][row].detach().cpu().numpy().astype(bool)
            target = batch["target"][row].detach().cpu().numpy()
            utilities = batch["utility"][row].detach().cpu().numpy()
            probs = probabilities[row]
            keys = batch["keys"][row]
            chosen = int(np.argmax(np.where(valid, probs, -np.inf)))
            chosen_key = keys[chosen]
            hit = float(chosen_key in batch["best"][row])
            confidence.append((float(probs[chosen]), hit, weight))
            predicted_counts[chosen_key] = predicted_counts.get(chosen_key, 0) + 1
            nll += weight * float(-(target[valid] * np.log(np.maximum(probs[valid], 1e-8))).sum())
            brier += weight * float(((probs[valid] - target[valid]) ** 2).sum())
            correct += weight * hit
            expected_utility += weight * float((probs[valid] * utilities[valid]).sum())
            oracle_utility += weight * float(utilities[valid].max())
            random_utility += weight * float(utilities[valid].mean())
            native_key = batch["native"][row]
            if native_key in keys:
                native_utility += weight * float(utilities[keys.index(native_key)])
            else:
                native_missing += 1
            total_weight += weight
    return {
        "records": int(len(records)),
        "weighted_records": total_weight,
        "nll": nll / max(total_weight, 1e-8),
        "brier": brier / max(total_weight, 1e-8),
        "ece": _ece(confidence, total_weight),
        "best_candidate_accuracy": correct / max(total_weight, 1e-8),
        "validation_utility": expected_utility / max(total_weight, 1e-8),
        "native_score_utility": native_utility / max(total_weight, 1e-8),
        "native_action_missing": native_missing,
        "oracle_utility": oracle_utility / max(total_weight, 1e-8),
        "random_utility": random_utility / max(total_weight, 1e-8),
        "predicted_candidate_distribution": predicted_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("candidate_mlp", "candidate_jev"), required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda:4")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.hidden_dim < 1 or args.lr <= 0:
        raise ValueError("epochs, batch-size, hidden-dim and lr must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_records = read_records(args.train.resolve())
    val_records = read_records(args.val.resolve())
    train_provenance = provenance(train_records, args.train.resolve())
    val_provenance = provenance(val_records, args.val.resolve())
    shared_provenance_keys = (
        "source_commit",
        "gmt_checkpoint_sha256",
        "annotations_sha256",
        "horizon",
    )
    if any(train_provenance[key] != val_provenance[key] for key in shared_provenance_keys):
        raise ValueError(f"train/val provenance mismatch: train={train_provenance} val={val_provenance}")
    train_sequences = sorted({str(record["sequence"]) for record in train_records})
    val_sequences = sorted({str(record["sequence"]) for record in val_records})
    if set(train_sequences) & set(val_sequences):
        raise ValueError("train/val sequence overlap")
    state_dim = len(train_records[0]["state_features"])
    device = torch.device(args.device)
    model = CandidateConditionedScorer(state_dim=state_dim, candidate_dim=8, hidden_dim=args.hidden_dim, model_name=args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(train_records))
        train_loss = 0.0
        train_weight = 0.0
        for start in range(0, len(order), args.batch_size):
            group = [train_records[int(index)] for index in order[start : start + args.batch_size]]
            loss, _output, batch, _losses = forward_batch(model, group, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_weight = float(batch["weights"].sum().item())
            train_loss += float(loss.item()) * batch_weight
            train_weight += batch_weight
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss / max(train_weight, 1e-8),
            "val": evaluate(model, val_records, device),
        })
        print(json.dumps(history[-1], sort_keys=True), flush=True)
    final_val = evaluate(model, val_records, device)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "model_name": args.model,
        "state_dim": state_dim,
        "candidate_dim": 8,
        "hidden_dim": args.hidden_dim,
        "seed": args.seed,
        "train_sequences": train_sequences,
        "val_sequences": val_sequences,
        "train_dataset": str(args.train.resolve()),
        "val_dataset": str(args.val.resolve()),
        "train_provenance": train_provenance,
        "val_provenance": val_provenance,
        "candidate_ids_used_as_features": False,
        "future_gt_used_by_model": False,
    }
    torch.save(checkpoint, args.output / "model.pth")
    report = {
        "status": "COMPLETE",
        "classification": "SCREENING_ONLY_NOT_FOR_FINAL_SELECTION_NOT_FOR_PAPER_RESULT",
        "model": args.model,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "device": str(device),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "train_sequences": train_sequences,
        "val_sequences": val_sequences,
        "provenance": train_provenance,
        "history": history,
        "validation": final_val,
        "candidate_ids_used_as_features": False,
        "future_gt_used_by_model": False,
        "full_h8_authorized": False,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
