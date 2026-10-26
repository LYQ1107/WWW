#!/usr/bin/env python3
"""Copy completed local scenes to NAS; reclaim only verified duplicate bytes."""
from datetime import datetime, timezone
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import time

from run_worldobject_v5_controller import FROZEN_SOURCE, read_json, write_json, process_identity
from worldobject_v5_durable_storage import (
    pinned_nas_root, digest, verified_copy, replace_local_with_verified_symlink)


def scene_has_live_worker(runtime, variant, scene):
    main = read_json(runtime / "controller_state.json", {}).get("jobs", {})
    auxiliary = read_json(runtime / "auxiliary_jobs.json", {})
    for job in [*main.values(), *auxiliary.values()]:
        command = job.get("command", [])
        if "--variant" not in command or "--scene" not in command:
            continue
        if (command[command.index("--variant") + 1] == variant
                and int(command[command.index("--scene") + 1]) == scene
                and process_identity(job["pid"]) == job.get("identity")
                and job.get("identity") is not None):
            return True
    return False


def archive_scene(folder, nas_root, variant, scene, copy_state_root=None):
    manifest = read_json(folder / "manifest.json")
    if (manifest["status"] != "COMPLETE" or manifest["variant"] != variant or manifest["scene"] != scene
            or manifest["v5_source_sha256"] != FROZEN_SOURCE or manifest["GT_in_inference"]
            or not manifest["causal"]):
        raise ValueError("ONLY_COMPLETE_FROZEN_CAUSAL_SCENES_CAN_BE_ARCHIVED")
    hashes = dict(manifest["output_sha256"])
    if set(hashes) != {"tracks.jsonl", "timings.jsonl", "birth_audit.jsonl"}:
        raise ValueError("FROZEN_OUTPUT_FILE_SET_REQUIRED")
    manifest_sha = digest(folder / "manifest.json")
    hashes["manifest.json"] = manifest_sha
    if (folder / "progress.json").is_file():
        hashes["progress.json"] = digest(folder / "progress.json")
    canonical = nas_root / f"predictions/{variant}/scene_{scene:02d}"
    copy_proofs = {}
    with pinned_nas_root(nas_root) as (remote, mount_identity):
        target = remote / f"predictions/{variant}/scene_{scene:02d}"
        target.parent.mkdir(parents=True, exist_ok=True)
        exists = target.exists()
        stage = target if exists else target.with_name(f".{target.name}.copying.{manifest_sha}")
        if exists and digest(target / "manifest.json") != manifest_sha:
            raise ValueError("EXISTING_NAS_SCENE_DIFFERS_NEVER_OVERWRITE")
        stage.mkdir(exist_ok=True)
        # Publish the complete manifest after its three output files.
        for name in sorted(hashes, key=lambda name: name == "manifest.json"):
            state_path = (copy_state_root / f"{variant}_scene{scene:02d}" / (name + ".json")) if copy_state_root else None
            copy_proofs[name] = verified_copy(folder / name, stage / name, hashes[name], state_path, str(canonical / name))
        if not exists:
            for name in hashes:
                (stage / name).chmod(0o444)
            if target.exists():
                raise FileExistsError("NAS scene appeared during copy; preserve both")
            stage.rename(target)
    return {"status": "COMPLETE_NAS_SCENE_OUTPUT_READBACK_VERIFIED", "variant": variant, "scene": scene,
            "local_directory": str(folder), "NAS_directory": str(canonical),
            "manifest_sha256": manifest_sha, "file_sha256": hashes,
            "per_file_remote_verification": copy_proofs,
            "verified_UTC": datetime.now(timezone.utc).isoformat(),
            "NAS_mount_identity_verified_before_and_after": mount_identity,
            "model_or_prediction_bytes_changed": False, "local_files_reclaimed": False}


def main():
    parser = argparse.ArgumentParser()
    for key in ("work", "runtime", "nas-root"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--interval", type=float, default=60.)
    parser.add_argument("--reclaim-below-gib", type=float, default=18.)
    parser.add_argument("--reclaim-all", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    lock = (args.runtime / "prediction_archive.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = args.runtime / "prediction_archive_status.json"
    state = read_json(state_path, {"scenes": {}})
    state.update({"pid": os.getpid(), "reclaim_below_GiB": args.reclaim_below_gib,
                  "running_workers_terminated": 0, "prediction_bytes_modified": False})
    while True:
        for variant in ("B0", "B1", "B2", "B3", "B4"):
            for scene in range(1, 23):
                folder = args.work / f"predictions/{variant}/scene_{scene:02d}"
                if folder.is_symlink() or not (folder / "manifest.json").is_file():
                    continue
                if scene_has_live_worker(args.runtime, variant, scene):
                    continue
                key = variant + "/" + str(scene)
                receipt_path = args.work / f"provenance/prediction_archive/{variant}_scene{scene:02d}.json"
                previous = read_json(receipt_path)
                reclaim = args.reclaim_all or shutil.disk_usage(args.work).free < args.reclaim_below_gib * 1024**3
                if previous and previous["manifest_sha256"] == digest(folder / "manifest.json") and not reclaim:
                    continue
                state.update({"status": "NAS_SCENE_COPY_PENDING_READBACK", "pending": key,
                              "pending_since_UTC": datetime.now(timezone.utc).isoformat()})
                write_json(state_path, state)
                try:
                    receipt = archive_scene(folder, args.nas_root, variant, scene,
                                            args.work / "provenance/prediction_copy_state")
                    write_json(receipt_path, receipt)
                    if reclaim:
                        if scene_has_live_worker(args.runtime, variant, scene):
                            raise ValueError("SCENE_WORKER_BECAME_LIVE_KEEP_LOCAL_COPY")
                        replace_local_with_verified_symlink(folder, receipt["NAS_directory"])
                        receipt["local_files_reclaimed"] = True
                        write_json(receipt_path, receipt)
                    state["scenes"][key] = receipt
                    state.update({"status": "VERIFIED_NAS_SCENE_ARCHIVES", "pending": None})
                    state.pop("last_error", None)
                    write_json(state_path, state)
                except (OSError, ValueError) as error:
                    state.update({"status": "ARCHIVE_RETRY_LOCAL_PREDICTIONS_PRESERVED",
                                  "last_error": {"type": type(error).__name__, "message": str(error),
                                                 "UTC": datetime.now(timezone.utc).isoformat()}})
                    write_json(state_path, state)
                    # A disconnect must not generate 110 futile NAS attempts.
                    break
            else:
                continue
            break
        if args.once:
            return
        time.sleep(max(30., args.interval))


if __name__ == "__main__":
    main()
