"""Stage-A training for the causal full-TRAIN association heads.

The native GMT score is the immutable base score.  Every learned model adds a
zero-initialized residual bounded by 0.25, so step zero is exactly the native
ranking.  Ω and DINO controls use the same listwise candidate groups and the
same number of optimizer updates; only the declared frozen evidence columns
change.  This script does not run GMT or open raw GT: the dataset builder has
already applied the locked post-audit masks.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "manifests/omega_full24_training_protocol_v1.json"


class BoundedResidual(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, max_residual: float = 0.25):
        super().__init__()
        self.max_residual = float(max_residual)
        self.body = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        # The output is exactly zero at initialization; the base native score
        # therefore remains the model's first-step prediction.
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x):
        return self.max_residual * torch.tanh(self.body(x).squeeze(-1))


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path):
    archive = np.load(path)
    arrays = {name: archive[name] for name in archive.files}
    offsets = arrays["offsets"]
    labels = arrays["labels"]
    group_ids = []
    for index in range(len(offsets) - 1):
        lo, hi = int(offsets[index]), int(offsets[index + 1])
        group_labels = labels[lo:hi]
        if len(group_labels) and group_labels.any():
            group_ids.append(index)
    arrays["group_ids"] = np.asarray(group_ids, dtype=np.int64)
    return arrays


def _normalizer(fit, indices):
    rows = []
    for index in indices:
        lo, hi = int(fit["offsets"][index]), int(fit["offsets"][index + 1])
        rows.append(np.asarray(fit["candidate_features"][lo:hi], dtype=np.float32))
    values = np.concatenate(rows, axis=0)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _batch(arrays, ids, device, input_indices, mean, std):
    lengths = [int(arrays["offsets"][index + 1] - arrays["offsets"][index]) for index in ids]
    width = max(lengths)
    feature_count = len(input_indices)
    features = np.zeros((len(ids), width, feature_count), dtype=np.float32)
    labels = np.zeros((len(ids), width), dtype=np.bool_)
    mask = np.zeros((len(ids), width), dtype=np.bool_)
    native = np.full((len(ids), width), -np.inf, dtype=np.float32)
    for row, index in enumerate(ids):
        lo, hi = int(arrays["offsets"][index]), int(arrays["offsets"][index + 1])
        value = np.asarray(arrays["candidate_features"][lo:hi], dtype=np.float32)
        value = (value - mean) / std
        count = hi - lo
        features[row, :count] = value[:, input_indices]
        labels[row, :count] = np.asarray(arrays["labels"][lo:hi], dtype=np.bool_)
        mask[row, :count] = True
        native[row, :count] = np.asarray(arrays["candidate_features"][lo:hi, 0], dtype=np.float32)
    return (
        torch.from_numpy(features).to(device),
        torch.from_numpy(labels).to(device),
        torch.from_numpy(mask).to(device),
        torch.from_numpy(native).to(device),
    )


def _loss(scores, labels, mask):
    valid_scores = scores.masked_fill(~mask, -torch.inf)
    positive_scores = scores.masked_fill(~(mask & labels), -torch.inf)
    return (torch.logsumexp(valid_scores, dim=1) - torch.logsumexp(positive_scores, dim=1)).mean()


@torch.no_grad()
def _evaluate(model, arrays, group_ids, device, input_indices, mean, std, batch_size):
    model.eval()
    losses = []
    total = native_correct = model_correct = fixes = harms = 0
    correctable_total = correctable_native = correctable_model = 0
    for start in range(0, len(group_ids), batch_size):
        ids = group_ids[start : start + batch_size].tolist()
        features, labels, mask, native = _batch(arrays, ids, device, input_indices, mean, std)
        scores = native + model(features)
        losses.append(float(_loss(scores, labels, mask)))
        model_top = scores.argmax(dim=1)
        native_top = native.argmax(dim=1)
        row = torch.arange(len(ids), device=device)
        model_good = labels[row, model_top]
        native_good = labels[row, native_top]
        total += len(ids)
        native_correct += int(native_good.sum())
        model_correct += int(model_good.sum())
        fixes += int((~native_good & model_good).sum())
        harms += int((native_good & ~model_good).sum())
        corr = torch.from_numpy(np.asarray(arrays["correctable"][ids], dtype=np.bool_)).to(device)
        correctable_total += int(corr.sum())
        correctable_native += int((corr & native_good).sum())
        correctable_model += int((corr & model_good).sum())
    return {
        "groups": total,
        "listwise_loss": float(np.mean(losses)) if losses else None,
        "native_top1": native_correct / total if total else None,
        "model_top1": model_correct / total if total else None,
        "fixes_vs_native": fixes,
        "harms_vs_native": harms,
        "net_fixes_vs_native": fixes - harms,
        "correctable_groups": correctable_total,
        "correctable_native_top1": correctable_native / correctable_total if correctable_total else None,
        "correctable_model_top1": correctable_model / correctable_total if correctable_total else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["native", "omega", "dino"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    if args.seed not in protocol["stages"]["A"]["seeds"]:
        raise ValueError("seed is outside the pre-registered Stage-A set")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fit = _load(args.dataset / "fit.npz")
    validation = _load(args.dataset / "validation.npz")
    if args.model == "native":
        input_indices = np.asarray([0, 1, 8, 9], dtype=np.int64)
    else:
        input_indices = np.arange(10, dtype=np.int64)
    mean, std = _normalizer(fit, fit["group_ids"])
    model = BoundedResidual(len(input_indices)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    fit_ids = fit["group_ids"]
    validation_ids = validation["group_ids"]
    epoch_targets = set(protocol["stages"]["A"]["epochs"])
    curves = []
    checkpoints = []
    started = time.perf_counter()
    for epoch in range(1, max(epoch_targets) + 1):
        rng = np.random.default_rng(args.seed + epoch)
        order = fit_ids.copy()
        rng.shuffle(order)
        model.train()
        train_losses = []
        for start in range(0, len(order), args.batch_size):
            ids = order[start : start + args.batch_size].tolist()
            features, labels, mask, native = _batch(fit, ids, device, input_indices, mean, std)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(native + model(features), labels, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        if epoch in epoch_targets:
            train_metrics = _evaluate(model, fit, fit_ids, device, input_indices, mean, std, args.batch_size)
            validation_metrics = _evaluate(model, validation, validation_ids, device, input_indices, mean, std, args.batch_size)
            curve = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "train": train_metrics, "validation": validation_metrics}
            curves.append(curve)
            checkpoint_path = args.output / f"stageA_{args.model}_seed{args.seed}_epoch{epoch}.pt"
            torch.save({
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "model": args.model, "seed": args.seed, "epoch": epoch,
                "input_indices": input_indices.tolist(), "normalizer_mean": mean, "normalizer_std": std,
                "max_residual": 0.25, "hidden_dim": 64, "dataset_manifest_sha256": _sha256(args.dataset / "manifest.json"),
                "protocol_sha256": _sha256(PROTOCOL), "foundation": "native GMT score plus frozen causal Ω evidence" if args.model == "omega" else "native GMT score only" if args.model == "native" else "native GMT score plus frozen DINO control evidence",
                "metrics": curve,
            }, checkpoint_path)
            checkpoints.append({"epoch": epoch, "path": str(checkpoint_path), "sha256": _sha256(checkpoint_path), "validation": validation_metrics})
            print(json.dumps({"model": args.model, "seed": args.seed, "epoch": epoch, "validation": validation_metrics}), flush=True)
    report = {
        "status": "COMPLETE_STAGE_A_HEAD_TRAINING",
        "model": args.model, "seed": args.seed, "fit_scene_ids": protocol["scene_split"]["fit_video_ids"],
        "validation_scene_ids": protocol["scene_split"]["validation_video_ids"], "epochs": sorted(epoch_targets),
        "input_indices": input_indices.tolist(), "max_residual": 0.25, "zero_initialized_residual": True,
        "dataset": str(args.dataset), "dataset_manifest_sha256": _sha256(args.dataset / "manifest.json"),
        "protocol_sha256": _sha256(PROTOCOL), "curves": curves, "checkpoints": checkpoints,
        "wall_seconds": time.perf_counter() - started,
        "selection_rule": "minimum validation listwise loss among pre-registered epochs; identity gate evaluated after all seeds",
        "test_accessed": False, "ground_truth_opened": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "model": args.model, "seed": args.seed, "seconds": report["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
