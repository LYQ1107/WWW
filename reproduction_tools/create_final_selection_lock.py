"""Create the immutable gate required before any official JEV test read."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CHECKPOINT_SHA256 = (
    "sha256:cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def require_canonical_checkpoint(path: Path) -> str:
    """Reject proxy checkpoints before creating an official selection lock."""
    digest = sha256_file(path)
    if digest != CANONICAL_CHECKPOINT_SHA256:
        raise ValueError(
            "official selection lock requires canonical model_20000 checkpoint "
            f"{CANONICAL_CHECKPOINT_SHA256}, got {digest} for {path}"
        )
    return digest


def current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def load_policy_split(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    for field in ("train_sequences", "val_sequences", "sequence_hash", "seed"):
        if field not in payload:
            raise ValueError(f"policy split is missing {field}")
    if payload.get("official_test_used_for_search") is not False:
        raise ValueError("policy split is not marked as official-test excluded")
    if payload.get("official_test_access") != "BLOCKED_BEFORE_FINAL_SELECTION_LOCK":
        raise ValueError("policy split does not carry the pre-lock TEST access boundary")
    if payload.get("official_test_files"):
        raise ValueError("policy split includes official TEST files")
    if not payload["train_sequences"] or not payload["val_sequences"]:
        raise ValueError("policy split must contain non-empty train and val sequences")
    train = set(payload["train_sequences"])
    val = set(payload["val_sequences"])
    if train & val:
        raise ValueError("policy split train/val overlap")
    return payload


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
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


def build_lock(args: argparse.Namespace) -> Mapping[str, Any]:
    checkpoint = args.checkpoint.resolve()
    split_path = args.policy_split.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha256 = require_canonical_checkpoint(checkpoint)
    split = load_policy_split(split_path)
    try:
        hyperparameters = json.loads(args.hyperparameters)
    except json.JSONDecodeError as exc:
        raise ValueError("--hyperparameters must be valid JSON") from exc
    if not isinstance(hyperparameters, Mapping):
        raise ValueError("--hyperparameters must decode to a JSON object")
    selection_manifest_arg = getattr(args, "selection_manifest", None)
    selection_manifest = (
        Path(selection_manifest_arg).resolve()
        if selection_manifest_arg is not None
        else None
    )
    selection_payload = None
    if selection_manifest is not None:
        if not selection_manifest.is_file():
            raise FileNotFoundError(selection_manifest)
        selection_payload = json.loads(selection_manifest.read_text(encoding="utf-8"))
        if not isinstance(selection_payload, Mapping) or selection_payload.get("status") != "PASS":
            raise ValueError("selection protocol manifest must have status=PASS")
    payload = {
        "schema_version": 1,
        "lock_type": "FINAL_SELECTION_LOCK",
        "selection_scope": "CANONICAL_MODEL20000_ONLY",
        "official_test_authority": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gmt_checkpoint": str(checkpoint),
        "gmt_checkpoint_sha256": checkpoint_sha256,
        "canonical_checkpoint_authority": True,
        "code_commit": args.code_commit or current_commit(),
        "state_schema_version": int(args.state_schema_version),
        "counterfactual_engine_version": args.counterfactual_engine_version,
        "utility_definition": args.utility_definition,
        "horizon": int(args.horizon),
        "selected_jev_architecture": args.jev_architecture,
        "selected_hidden_size": int(args.hidden_size),
        "selected_calibration_method": args.calibration_method,
        "selected_threshold_baseline": args.threshold_baseline,
        "selected_mlp_baseline": args.mlp_baseline,
        "hyperparameters": dict(hyperparameters),
        "policy_split": str(split_path),
        "policy_split_sha256": sha256_file(split_path),
        "policy_split_sequence_hash": split["sequence_hash"],
        "official_test_gate": {
            "official_test_read_allowed_after_lock": True,
            "lock_created_before_official_test": True,
            "selection_scope": "CANONICAL_MODEL20000_ONLY",
            "official_test_authority": True,
        },
    }
    if selection_manifest is not None:
        payload["selection_protocol"] = str(selection_manifest)
        payload["selection_protocol_sha256"] = sha256_file(selection_manifest)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("manifests/FINAL_SELECTION_LOCK.json"))
    parser.add_argument("--state-schema-version", type=int, required=True)
    parser.add_argument("--counterfactual-engine-version", required=True)
    parser.add_argument("--utility-definition", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--jev-architecture", required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--calibration-method", required=True)
    parser.add_argument("--threshold-baseline", required=True)
    parser.add_argument("--mlp-baseline", required=True)
    parser.add_argument("--hyperparameters", default="{}")
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--code-commit")
    args = parser.parse_args()
    if args.horizon < 1 or args.hidden_size < 1 or args.state_schema_version < 1:
        raise ValueError("horizon, hidden-size, and state-schema-version must be positive")
    lock = build_lock(args)
    atomic_write_json(args.output, lock)
    print(json.dumps({"status": "PASS", "output": str(args.output), "checkpoint_sha256": lock["gmt_checkpoint_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
