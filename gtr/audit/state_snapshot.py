"""Serializable snapshots for the mutable GMT association state.

Snapshots are audit infrastructure only.  They include the Python, NumPy,
Torch CPU/CUDA RNG streams and every tracker container that is mutated by the
released association loop, so an event branch can be restored to exactly the
same pre-event state.
"""
from __future__ import annotations

import copy
import hashlib
import pickle
import random
from typing import Any

import numpy as np
import torch

from ..modeling.meta_arch.gtr_rcnn import old_ids, old_reids, poss_ids


def _clone(value):
    return copy.deepcopy(value)


def capture_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [x.clone() for x in torch.cuda.get_rng_state_all()]
    else:
        state["torch_cuda"] = []
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].detach().cpu().clone())
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all([x.detach().cpu().clone() for x in state["torch_cuda"]])


def snapshot_tracker_state(
    *,
    frame_index: int,
    view_index: int,
    instances,
    id_count: int,
    id_count_dict: dict,
    id_reid_dict: dict,
    window_start: int | None = None,
    window_end: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture all mutable local and module-global GMT tracker state."""
    return {
        "format": "gmt-tracker-state-snapshot-v1",
        "frame_index": int(frame_index),
        "view_index": int(view_index),
        "window_start": None if window_start is None else int(window_start),
        "window_end": None if window_end is None else int(window_end),
        "instances": _clone(instances),
        "id_count": int(id_count),
        "id_count_dict": _clone(id_count_dict),
        "id_reid_dict": _clone(id_reid_dict),
        "poss_ids": _clone(poss_ids.poss_ids),
        "old_ids": _clone(old_ids.old_ids),
        "old_reids": _clone(old_reids.old_reids),
        "rng": capture_rng_state(),
        "extra": _clone(extra or {}),
    }


def restore_tracker_state(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Restore module-global state and return cloned local state."""
    if snapshot.get("format") != "gmt-tracker-state-snapshot-v1":
        raise ValueError("unsupported GMT tracker snapshot format")
    poss_ids.poss_ids = _clone(snapshot["poss_ids"])
    old_ids.old_ids = _clone(snapshot["old_ids"])
    old_reids.old_reids = _clone(snapshot["old_reids"])
    restore_rng_state(snapshot["rng"])
    return {
        "frame_index": int(snapshot["frame_index"]),
        "view_index": int(snapshot["view_index"]),
        "window_start": snapshot.get("window_start"),
        "window_end": snapshot.get("window_end"),
        "instances": _clone(snapshot["instances"]),
        "id_count": int(snapshot["id_count"]),
        "id_count_dict": _clone(snapshot["id_count_dict"]),
        "id_reid_dict": _clone(snapshot["id_reid_dict"]),
        "extra": _clone(snapshot.get("extra", {})),
    }


def state_digest(value: Any) -> str:
    """Stable digest for a snapshot or restored state (including tensors)."""
    h = hashlib.sha256()

    def visit(x):
        if torch.is_tensor(x):
            y = x.detach().cpu().contiguous()
            h.update(b"tensor")
            h.update(str(y.dtype).encode())
            h.update(repr(tuple(y.shape)).encode())
            h.update(y.numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(b"ndarray")
            h.update(str(x.dtype).encode())
            h.update(repr(tuple(x.shape)).encode())
            h.update(x.tobytes())
        elif isinstance(x, dict):
            h.update(b"dict")
            for k in sorted(x, key=lambda y: repr(y)):
                visit(k); visit(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(type(x).__name__.encode())
            for y in x: visit(y)
        elif isinstance(x, set):
            h.update(b"set")
            for y in sorted(x, key=repr): visit(y)
        elif hasattr(x, "get_fields") and hasattr(x, "image_size"):
            h.update(b"Instances")
            visit(tuple(x.image_size))
            fields = x.get_fields()
            for k in sorted(fields):
                visit(k); visit(fields[k])
        elif hasattr(x, "tensor"):
            h.update(type(x).__name__.encode()); visit(x.tensor)
        else:
            try:
                h.update(pickle.dumps(x, protocol=4))
            except Exception:
                h.update(repr(x).encode())

    visit(value)
    return h.hexdigest()


def save_snapshot(path, snapshot: dict[str, Any]) -> None:
    """Atomically serialize one snapshot to a local ``.pt`` file."""
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(snapshot, tmp)
    tmp.replace(path)


def load_snapshot(path):
    return torch.load(path, map_location="cpu")
