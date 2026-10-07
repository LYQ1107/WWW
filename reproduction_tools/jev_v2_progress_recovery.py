"""Progress, append, and checkpoint primitives for future JEV v2 chunk runs.

This module is intentionally not imported by the currently running video01
builder.  It provides the recovery boundary for a future frozen-worktree run:
JSON progress is atomically replaced, records are fsync'ed as they are
appended, and a checkpoint contains the mutable GMT state, Python trajectory
RNG state, and exact provenance needed to reject an incompatible resume.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch

from jev_counterfactual_v2 import MutableGMTState, _state_signature


PROGRESS_SCHEMA = "jev_v2_progress_v1"
CHECKPOINT_SCHEMA = "jev_v2_state_checkpoint_v1"


@dataclass(frozen=True)
class Progress:
    frame: int
    view: int
    event_order: int
    completed_events: int
    total_events: int
    branch_count: int
    elapsed_seconds: float
    last_update_utc: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = PROGRESS_SCHEMA
        return payload


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def state_signature_sha256(state: MutableGMTState) -> str:
    encoded = json.dumps(
        _state_signature(state),
        sort_keys=True,
        separators=(",", ":"),
        default=repr,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_progress(path: Path, progress: Progress) -> None:
    """Atomically publish the latest progress record."""

    if progress.completed_events < 0 or progress.total_events < 0:
        raise ValueError("event counts must be non-negative")
    if progress.completed_events > progress.total_events:
        raise ValueError("completed_events cannot exceed total_events")
    if progress.branch_count < 0:
        raise ValueError("branch_count must be non-negative")
    _atomic_json_write(path, progress.as_dict())


def append_record(path: Path, record: Mapping[str, Any]) -> None:
    """Append one record and force it to stable storage before returning."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def save_checkpoint(
    path: Path,
    *,
    state: MutableGMTState,
    provenance: Mapping[str, Any],
    progress: Progress,
) -> None:
    """Atomically save mutable state plus resume provenance."""

    if state.trajectory_rng_state is None or state.trajectory_rng_seed is None:
        raise ValueError("checkpoint requires explicit trajectory RNG provenance")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "provenance": dict(provenance),
        "progress": progress.as_dict(),
        "state_signature_sha256": state_signature_sha256(state),
        "state": state,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    try:
        torch.save(payload, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(
    path: Path,
    *,
    expected_provenance: Mapping[str, Any],
) -> tuple[MutableGMTState, dict[str, Any], dict[str, Any]]:
    """Load a checkpoint and fail closed on source/input provenance drift."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise RuntimeError("unsupported JEV v2 checkpoint schema")
    actual = dict(payload.get("provenance", {}))
    expected = dict(expected_provenance)
    if actual != expected:
        raise RuntimeError(
            "checkpoint provenance mismatch; refusing to resume with different "
            f"inputs/source: expected={expected!r} actual={actual!r}"
        )
    state = payload.get("state")
    # Some launchers expose ``reproduction_tools`` as a package while the
    # production scripts import the same file as the top-level module.  The
    # two import spellings can create distinct Python class identities even
    # though the serialized state is the same protocol object.
    if not isinstance(state, MutableGMTState) and state.__class__.__name__ != "MutableGMTState":
        raise RuntimeError("checkpoint does not contain MutableGMTState")
    if state.trajectory_rng_state is None or state.trajectory_rng_seed is None:
        raise RuntimeError("checkpoint lacks trajectory RNG state")
    if payload.get("state_signature_sha256") != state_signature_sha256(state):
        raise RuntimeError("checkpoint state signature does not validate")
    return state, dict(payload["progress"]), actual


__all__ = [
    "CHECKPOINT_SCHEMA",
    "PROGRESS_SCHEMA",
    "Progress",
    "append_record",
    "load_checkpoint",
    "save_checkpoint",
    "state_signature_sha256",
    "utc_now",
    "write_progress",
]
