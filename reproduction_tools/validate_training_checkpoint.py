"""Validate a completed GMT training checkpoint before stage transition."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_tensors(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(finite_tensors(item) for item in value.values())
    if torch.is_tensor(value):
        if not (value.is_floating_point() or value.is_complex()):
            return True
        return bool(torch.isfinite(value).all().item())
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-iteration", type=int, required=True)
    parser.add_argument("--expected-scheduler-iteration", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    expected_scheduler_iteration = (
        args.expected_scheduler_iteration
        if args.expected_scheduler_iteration is not None
        else args.expected_iteration
    )
    if output.exists():
        raise RuntimeError(f"refusing to overwrite validation report: {output}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    payload = torch.load(str(checkpoint), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    model = payload.get("model")
    optimizer = payload.get("optimizer")
    scheduler = payload.get("scheduler")
    iteration = int(payload.get("iteration", -1))
    if not isinstance(model, Mapping):
        raise ValueError("checkpoint has no model state")
    if not isinstance(optimizer, Mapping) or not isinstance(scheduler, Mapping):
        raise ValueError("checkpoint is missing optimizer/scheduler state")
    scheduler_epoch = int(scheduler.get("last_epoch", -1))
    if iteration != args.expected_iteration:
        raise ValueError(
            f"checkpoint iteration {iteration} != expected {args.expected_iteration}"
        )
    if scheduler_epoch != expected_scheduler_iteration:
        raise ValueError(
            f"scheduler last_epoch {scheduler_epoch} != expected {expected_scheduler_iteration}"
        )
    if not finite_tensors(model):
        raise ValueError("model contains non-finite tensors")
    report = {
        "status": "PASS",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "bytes": checkpoint.stat().st_size,
        "iteration": iteration,
        "scheduler_last_epoch": scheduler_epoch,
        "global_iteration": expected_scheduler_iteration,
        "model_keys": len(model),
        "model_finiteness": "PASS",
        "optimizer_state": "PRESENT",
        "checkpoint_reload": "PASS",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
