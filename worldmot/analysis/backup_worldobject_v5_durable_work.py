#!/usr/bin/env python3
"""Back up verified local scene checkpoints without involving model workers.

The original NAS run is an archive. Recovery snapshots use a separate subtree,
and a stalled network copy never blocks a trainer or reports a false success.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import time

from run_worldobject_v5_controller import FROZEN_SOURCE, read_json, write_json
from worldobject_v5_durable_storage import pinned_nas_root


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def checkpoint_files(work, variant, phase):
    """Read an immutable checkpoint selected by its atomic resume pointer."""
    base = work / "train" / variant / phase
    pointer = (base / "RESUME.json").read_bytes()
    info = json.loads(pointer)
    if info["v5_source_sha256"] != FROZEN_SOURCE or info["resume_boundary"] != "COMPLETE_SCENE_ONLY":
        raise ValueError("REGISTERED_COMPLETE_SCENE_CHECKPOINT_REQUIRED")
    name = info["checkpoint"]
    if Path(name).name != name:
        raise ValueError("CHECKPOINT_MUST_BE_INSIDE_PHASE_DIRECTORY")
    contents = (base / name).read_bytes()
    if sha_bytes(contents) != info["sha256"]:
        raise ValueError("CHECKPOINT_BACKUP_SHA_MISMATCH")
    files = {f"train/{variant}/{phase}/{name}": contents,
             f"train/{variant}/{phase}/RESUME.json": pointer}
    for path in sorted(base.rglob("*")):
        if path.is_symlink() or not path.is_file() or path.suffix not in {".json", ".jsonl"}:
            continue
        if path.name == "RESUME.json" or ".partial" in path.name:
            continue
        files[str(path.relative_to(work))] = path.read_bytes()
    # Scope/auth files are private and remain only in the private NAS snapshot.
    for path in sorted((work / "provenance").glob("*.json")):
        if path.is_file() and ".partial" not in path.name:
            files[str(path.relative_to(work))] = path.read_bytes()
    gate = work / "train" / variant / "train_gate.json"
    if phase == "final" and gate.is_file():
        metadata = json.loads(gate.read_text())
        weights = Path(metadata["checkpoint"])
        # Final gate layout is checked by its strict model loader separately.
        if not weights.is_absolute():
            weights = gate.parent / weights
        payload = weights.read_bytes()
        if sha_bytes(payload) != metadata["checkpoint_sha256"]:
            raise ValueError("FINAL_WEIGHT_BACKUP_SHA_MISMATCH")
        files[str(gate.relative_to(work))] = gate.read_bytes()
        files[str(weights.relative_to(work))] = payload
    manifest = {"schema": "v5_durable_scene_backup_v1", "variant": variant, "phase": phase,
                "checkpoint_sha256": info["sha256"], "source_sha256": FROZEN_SOURCE,
                "scope_sha256": info["scope_sha256"],
                "resume_pointer_sha256": sha_bytes(pointer),
                "captured_UTC": datetime.now(timezone.utc).isoformat(),
                "files": {key: {"sha256": sha_bytes(value), "bytes": len(value)} for key, value in files.items()},
                "metadata_may_advance_after_checkpoint_capture": True,
                "authoritative_resume_state_is_checkpoint_payload": True}
    files["BACKUP_MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    return info, files, manifest


def make_archive(files):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, contents in sorted(files.items()):
            item = tarfile.TarInfo(name)
            item.size = len(contents)
            item.mode = 0o600
            archive.addfile(item, io.BytesIO(contents))
    return output.getvalue()


def publish_snapshot(destination, name, contents):
    """No overwrite; verify the remote bytes before returning a receipt."""
    target = destination / name
    destination.mkdir(parents=True, exist_ok=True)
    expected = sha_bytes(contents)
    if not target.exists():
        temporary = target.with_name(target.name + ".partial." + str(os.getpid()))
        with temporary.open("xb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)  # Atomic publication that refuses overwrite.
        temporary.unlink()
    actual = sha_bytes(target.read_bytes())
    if actual != expected:
        raise ValueError("NAS_BACKUP_READBACK_SHA_MISMATCH")
    return {"path": str(target), "bytes": len(contents), "sha256": actual,
            "verified_UTC": datetime.now(timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser()
    for name in ("work", "runtime", "nas-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--interval", type=float, default=300.)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    lock = (args.runtime / "nas_backup.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = args.runtime / "nas_backup_status.json"
    state = read_json(state_path, {"snapshots": {}})
    state.update({"pid": os.getpid(), "working_root": str(args.work),
                  "NAS_root": str(args.nas_root), "model_processes_modified": False})
    spool = args.runtime / "nas_backup_spool"
    spool.mkdir(exist_ok=True)
    while True:
        for variant in ("B3", "B4", "B2"):
            for phase in ("selection", "final"):
                try:
                    base = args.work / "train" / variant / phase
                    if not (base / "RESUME.json").exists():
                        continue
                    pointer = read_json(base / "RESUME.json")
                    key = variant + "_" + phase
                    gate_path = args.work / "train" / variant / "train_gate.json"
                    gate_sha = sha_bytes(gate_path.read_bytes()) if phase == "final" and gate_path.exists() else None
                    previous = state["snapshots"].get(key, {})
                    if (previous.get("checkpoint_sha256") == pointer["sha256"]
                            and previous.get("final_gate_sha256") == gate_sha):
                        continue
                    info, files, manifest = checkpoint_files(args.work, variant, phase)
                    contents = make_archive(files)
                    name = f"{variant}_{phase}_{info['checkpoint'][:-3]}_{info['sha256']}_{sha_bytes(contents)[:16]}.tar.gz"
                    local = spool / name
                    local.write_bytes(contents)
                    state.update({"status": "NAS_COPY_PENDING_READBACK", "pending": str(local),
                                  "pending_since_UTC": datetime.now(timezone.utc).isoformat()})
                    write_json(state_path, state)
                    with pinned_nas_root(args.nas_root) as (remote, mount_identity):
                        receipt = publish_snapshot(remote / "durable_recovery_snapshots", name, contents)
                    receipt.update({"path": str(args.nas_root / "durable_recovery_snapshots" / name),
                                    "NAS_mount_identity_verified_before_and_after": mount_identity})
                    state["snapshots"][key] = {**receipt, "checkpoint_sha256": info["sha256"],
                                               "final_gate_sha256": gate_sha, "manifest": manifest}
                    state.update({"status": "VERIFIED_NAS_SNAPSHOTS", "pending": None})
                    state.pop("last_error", None)
                    write_json(state_path, state)
                    # Keep only pending archives locally; durable originals remain.
                    local.unlink()
                except (OSError, ValueError) as error:
                    state.update({"status": "BACKUP_RETRY_LOCAL_TRAINING_UNAFFECTED",
                                  "last_error": {"type": type(error).__name__, "message": str(error),
                                                 "UTC": datetime.now(timezone.utc).isoformat()}})
                    write_json(state_path, state)
        if args.once:
            return
        time.sleep(max(30., args.interval))


if __name__ == "__main__":
    main()
