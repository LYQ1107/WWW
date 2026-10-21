"""Stage-B all-24 refit after the locked Stage-A epoch selection."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from train_omega_full24_heads import BoundedResidual, _batch, _evaluate, _load, _loss, _normalizer, _sha256


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "manifests/omega_full24_training_protocol_v1.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["native", "omega", "dino"], required=True)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--stage-a", type=Path, required=True)
    parser.add_argument("--selected-epoch", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    if args.seed != protocol["stages"]["B"]["seeds"][0] or args.selected_epoch not in protocol["stages"]["A"]["epochs"]:
        raise ValueError("Stage-B seed/epoch is outside the locked protocol")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data = _load(args.dataset / "all24.npz")
    ids = data["group_ids"]
    input_indices = np.asarray([0, 1, 8, 9] if args.model == "native" else list(range(10)), dtype=np.int64)
    mean, std = _normalizer(data, ids)
    model = BoundedResidual(len(input_indices)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    started = time.perf_counter()
    for epoch in range(1, args.selected_epoch + 1):
        rng = np.random.default_rng(args.seed + epoch)
        order = ids.copy()
        rng.shuffle(order)
        model.train()
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch_ids = order[start : start + args.batch_size].tolist()
            features, labels, mask, native = _batch(data, batch_ids, device, input_indices, mean, std)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(native + model(features), labels, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite Stage-B loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        print(json.dumps({"model": args.model, "epoch": epoch, "train_loss": float(np.mean(losses))}), flush=True)
    metrics = _evaluate(model, data, ids, device, input_indices, mean, std, args.batch_size)
    checkpoint = args.output / f"stageB_{args.model}_seed{args.seed}_epoch{args.selected_epoch}.pt"
    torch.save({
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model": args.model, "seed": args.seed, "epoch": args.selected_epoch,
        "input_indices": input_indices.tolist(), "normalizer_mean": mean, "normalizer_std": std,
        "max_residual": 0.25, "hidden_dim": 64, "dataset_manifest_sha256": _sha256(args.dataset / "manifest.json"),
        "stage_a_manifest_sha256": _sha256(args.stage_a), "protocol_sha256": _sha256(PROTOCOL), "metrics": metrics,
    }, checkpoint)
    report = {
        "status": "COMPLETE_STAGE_B_REFIT_ALL24_TRAIN",
        "model": args.model, "seed": args.seed, "selected_epoch": args.selected_epoch,
        "dataset": str(args.dataset), "dataset_manifest_sha256": _sha256(args.dataset / "manifest.json"),
        "stage_a_manifest": str(args.stage_a), "stage_a_manifest_sha256": _sha256(args.stage_a),
        "protocol_sha256": _sha256(PROTOCOL), "metrics_all24": metrics,
        "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
        "zero_initialized_residual": True, "max_residual": 0.25, "test_accessed": False,
        "wall_seconds": time.perf_counter() - started,
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "model": args.model, "metrics_all24": metrics}), flush=True)


if __name__ == "__main__":
    main()
