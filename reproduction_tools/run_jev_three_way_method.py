"""Train one locked H=8 comparison method for the first single-seed round."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

import torch

from train_jev import choose_model


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/liuyeqiang/anaconda3/envs/GMT/bin/python"
# The first decision gate is intentionally speed-only.  Multi-seed robustness
# is a later protocol phase and must not be mixed into this result.
SEEDS = (20261003,)
METHODS = {
    "question_threshold": "Learnable Threshold",
    "question_conditioned_mlp": "Generic MLP",
    "jev": "Full JEV",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, Mapping) else None


def archive(path: Path) -> None:
    if not path.exists():
        return
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = path.with_name(path.name + ".incomplete." + stamp)
    suffix = 1
    while destination.exists():
        destination = path.with_name(path.name + f".incomplete.{stamp}.{suffix}")
        suffix += 1
    path.rename(destination)


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(
        [
            str(ROOT),
            str(ROOT / "third_party/CenterNet2"),
            str(ROOT / "reproduction_tools"),
            env.get("PYTHONPATH", ""),
        ]
    )
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    return env


def run(command: list[str | Path], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(str(item) for item in command) + "\n")
        handle.flush()
        result = subprocess.run(
            [str(item) for item in command],
            cwd=ROOT,
            env=environment(),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {command}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(METHODS), required=True)
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    split = args.split_manifest.resolve()
    output = args.output.resolve()
    if not (dataset / "manifest.json").is_file() or not split.is_file():
        raise FileNotFoundError("shared dataset and policy split are required")
    if args.hidden_dim < 1 or args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError("training parameters must be positive")

    output.mkdir(parents=True, exist_ok=True)
    members = []
    for seed in SEEDS:
        run_root = output / f"seed_{seed}"
        metrics_path = run_root / "metrics.json"
        checkpoint = run_root / "model.pth"
        if (read_json(metrics_path) or {}).get("status") != "COMPLETE" or not checkpoint.is_file():
            if run_root.exists():
                archive(run_root)
            run(
                [
                    PYTHON,
                    "-u",
                    ROOT / "reproduction_tools/train_jev_compact.py",
                    "--dataset",
                    dataset,
                    "--output",
                    run_root,
                    "--model",
                    args.model,
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--hidden-dim",
                    str(args.hidden_dim),
                    "--lr",
                    str(args.lr),
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                    "--split-manifest",
                    split,
                ],
                run_root / "train.log",
            )
        metrics = read_json(metrics_path)
        if not metrics or metrics.get("status") != "COMPLETE" or not checkpoint.is_file():
            raise RuntimeError(f"incomplete training result: {run_root}")

        calibration_root = run_root / "calibration"
        calibration_report = calibration_root / "calibration_val_only.json"
        calibrated = calibration_root / "model_calibrated.pth"
        if (read_json(calibration_report) or {}).get("status") != "PASS" or not calibrated.is_file():
            if calibration_root.exists():
                archive(calibration_root)
            run(
                [
                    PYTHON,
                    "-u",
                    ROOT / "reproduction_tools/calibrate_jev_compact.py",
                    "--dataset",
                    dataset,
                    "--checkpoint",
                    checkpoint,
                    "--policy-split",
                    split,
                    "--output",
                    calibration_root,
                ],
                run_root / "calibration.log",
            )
        calibration = read_json(calibration_report)
        if not calibration or calibration.get("status") != "PASS" or not calibrated.is_file():
            raise RuntimeError(f"incomplete calibration result: {run_root}")

        model = choose_model(args.model, int(metrics["state_dim"]), args.hidden_dim)
        params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        members.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "calibrated_checkpoint": str(calibrated),
                "calibrated_checkpoint_sha256": sha256(calibrated),
                "train_metrics": str(metrics_path),
                "calibration_report": str(calibration_report),
                "temperature": float(calibration["temperature"]),
                "trainable_params": int(params),
                "validation": dict(metrics["history"][-1]["val"]),
                "validation_calibrated": dict(calibration["after"]),
            }
        )

    manifest = {
        "status": "PASS",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": args.model,
        "method_label": METHODS[args.model],
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset / "manifest.json"),
        "policy_split": str(split),
        "policy_split_sha256": sha256(split),
        "horizon": 8,
        "seeds": list(SEEDS),
        "seed_policy": "TEMPORARILY_DISABLED_MULTI_SEED",
        "multi_seed_robustness": "deferred_until_after_first_round_gate",
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "AdamW",
            "learning_rate": args.lr,
            "lr_search": "locked single value shared by all methods",
            "device": args.device,
            "hidden_dim": args.hidden_dim,
        },
        "calibration": {
            "method": "temperature_scaling_policy_val_only",
            "applied_to": "all_three_methods",
        },
        "members": members,
    }
    (output / "method_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "method": args.model, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
