"""Reproducible training/checkpoint helpers."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def rng_state() -> dict:
    # Store NumPy's array as a tensor so weights_only=True remains safe.
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": {
        "algorithm": state[0], "keys": torch.from_numpy(state[1].astype(np.int64)),
        "position": state[2], "gaussian": state[3], "cached_gaussian": state[4]},
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng_state(saved: dict) -> None:
    random.setstate(saved["python"])
    state = saved["numpy"]
    np.random.set_state((state["algorithm"], state["keys"].cpu().numpy().astype(np.uint32),
                         state["position"], state["gaussian"], state["cached_gaussian"]))
    torch.set_rng_state(saved["torch"].cpu())
    if saved["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda"]])


def save_checkpoint(path: str | Path, model, optimizer, epoch: int, manifest: dict) -> str:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": int(epoch), "manifest": manifest, "rng_state": rng_state()}
    torch.save(payload, p)
    return str(p)


def load_checkpoint(path: str | Path, model, optimizer=None, map_location="cpu") -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
