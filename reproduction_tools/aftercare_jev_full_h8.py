"""Wait for the formal H=8 build and run its audited CPU-only aftercare.

This process is intentionally separate from the GPU scheduler.  It does not
touch an incomplete shard, does not overwrite an existing artifact, never
reads official TEST data, and never starts policy training.  Once the queue
and all 24 official per-video manifests are COMPLETE, it runs the finalizer,
compact conversion, and fixed TRAIN-only policy split in that order.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping


PYTHON = "/home/liuyeqiang/anaconda3/envs/GMT/bin/python"
SEED = 20261003


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


def derived_paths(runtime: Path) -> dict[str, Path]:
    return {
        "partition": runtime / "partition",
        "shards": runtime / "formal_h8_full",
        "records": runtime / "FULL_H8_FORMAL.records.jsonl",
        "compact": runtime / "FULL_H8_POLICY_COMPACT",
        "split": runtime / "POLICY_SPLIT.json",
        "status": runtime / "POSTPROCESS_MANIFEST.json",
        "log": runtime / "postprocess.log",
    }


def official_shards_ready(paths: Mapping[str, Path], expected: Mapping[str, Any]) -> bool:
    for video_key in expected:
        video_root = paths["shards"] / f"video_{int(video_key):02d}"
        records = video_root / "records.jsonl"
        manifest = video_root / "manifest.json"
        if not records.is_file() or not manifest.is_file():
            return False
        try:
            payload = read_json(manifest)
        except (OSError, ValueError, TypeError):
            # A worker may be between the atomic rename and the next queue
            # heartbeat.  Treat that as not ready and poll again.
            return False
        if payload.get("status") != "COMPLETE":
            return False
        if int(payload.get("video_id", -1)) != int(video_key):
            return False
    return True


def queue_ready(queue: Path) -> tuple[bool, str]:
    if not queue.is_file():
        return False, "queue_missing"
    payload = read_json(queue)
    statuses = [str(item.get("status")) for item in (payload.get("videos") or {}).values()]
    if any(status == "FAILED" for status in statuses) or payload.get("status") == "FAILED":
        raise RuntimeError("formal H=8 queue contains FAILED video(s)")
    if not statuses or not all(status == "COMPLETE" for status in statuses):
        return False, "videos_not_complete"
    if payload.get("status") != "COMPLETE":
        return False, "scheduler_not_finalized"
    return True, "queue_complete"


def environment(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = ":".join(
        [
            str(root),
            str(root / "third_party/CenterNet2"),
            str(root / "reproduction_tools"),
            env.get("PYTHONPATH", ""),
        ]
    )
    return env


def run_step(name: str, command: list[Path | str], log: Path, root: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc_now()}] {name}: $ {' '.join(str(item) for item in command)}\n")
        handle.flush()
        result = subprocess.run(
            [str(item) for item in command],
            cwd=root,
            env=environment(root),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"{name} failed with return code {result.returncode}")


def update_status(path: Path, status: str, **fields: Any) -> None:
    current: dict[str, Any] = {}
    if path.is_file():
        try:
            current = dict(read_json(path))
        except (OSError, ValueError):
            current = {}
    current.update({"status": status, "updated_utc": utc_now(), **fields})
    atomic_json(path, current)


def validate_existing(paths: Mapping[str, Path], expected_records: int) -> dict[str, bool]:
    ready = {"finalize": False, "compact": False, "split": False}
    records_manifest = paths["records"].with_suffix(paths["records"].suffix + ".manifest.json")
    if paths["records"].is_file() or records_manifest.is_file():
        if not paths["records"].is_file() or not records_manifest.is_file():
            raise RuntimeError("partial finalizer output exists; refusing overwrite")
        payload = read_json(records_manifest)
        if payload.get("status") != "PASS" or int(payload.get("record_count", -1)) != expected_records:
            raise RuntimeError("existing finalizer output is not a matching PASS artifact")
        ready["finalize"] = True

    if paths["compact"].exists():
        manifest_path = paths["compact"] / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("partial compact dataset exists; refusing overwrite")
        payload = read_json(manifest_path)
        if payload.get("status") != "PASS" or int(payload.get("records", -1)) != expected_records:
            raise RuntimeError("existing compact dataset is not a matching PASS artifact")
        ready["compact"] = True

    if paths["split"].exists():
        payload = read_json(paths["split"])
        if int(payload.get("seed", -1)) != SEED or payload.get("official_test_used_for_search") is not False:
            raise RuntimeError("existing policy split is not the locked TRAIN-only seed split")
        ready["split"] = True
    return ready


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args()
    runtime = args.runtime_root.resolve()
    paths = derived_paths(runtime)
    root = Path(__file__).resolve().parents[1]
    queue = runtime / "queue_state.json"
    partition_manifest_path = paths["partition"] / "partition_manifest.json"
    if not partition_manifest_path.is_file():
        raise FileNotFoundError(partition_manifest_path)
    partition = read_json(partition_manifest_path)
    expected_videos = partition.get("videos") or {}
    expected_records = int(partition.get("total_main_decisions", -1))
    if partition.get("status") != "COMPLETE" or len(expected_videos) != 24 or expected_records <= 0:
        raise ValueError("formal partition manifest is not the locked 24-video H=8 manifest")
    runtime.mkdir(parents=True, exist_ok=True)
    update_status(
        paths["status"],
        "WAITING",
        runtime_root=str(runtime),
        partition_manifest_sha256=sha256(partition_manifest_path),
        expected_video_count=len(expected_videos),
        expected_records=expected_records,
        seed=SEED,
        official_test_read=False,
        gpu_training_started=False,
    )
    try:
        while True:
            ready, reason = queue_ready(queue)
            shard_ready = official_shards_ready(paths, expected_videos)
            update_status(paths["status"], "WAITING", queue_reason=reason, official_shards_ready=shard_ready)
            if ready and shard_ready:
                break
            time.sleep(max(5.0, float(args.poll_seconds)))

        existing = validate_existing(paths, expected_records)
        update_status(paths["status"], "RUNNING", queue_reason="queue_complete", official_shards_ready=True)
        if not existing["finalize"]:
            run_step(
                "finalize",
                [
                    PYTHON,
                    "-u",
                    root / "reproduction_tools/finalize_jev_full_h8.py",
                    "--shard-root",
                    paths["shards"],
                    "--partition-root",
                    paths["partition"],
                    "--output",
                    paths["records"],
                ],
                paths["log"],
                root,
            )
        update_status(paths["status"], "RUNNING", finalize="PASS")
        if not existing["compact"]:
            run_step(
                "compact",
                [
                    PYTHON,
                    "-u",
                    root / "reproduction_tools/jev_compact_dataset.py",
                    "--input",
                    paths["records"],
                    "--output",
                    paths["compact"],
                ],
                paths["log"],
                root,
            )
        update_status(paths["status"], "RUNNING", finalize="PASS", compact="PASS")
        if not existing["split"]:
            run_step(
                "policy_split",
                [
                    PYTHON,
                    "-u",
                    root / "reproduction_tools/create_jev_policy_split.py",
                    "--input",
                    paths["records"],
                    "--output",
                    paths["split"],
                    "--seed",
                    str(SEED),
                    "--val-fraction",
                    "0.2",
                ],
                paths["log"],
                root,
            )
        update_status(
            paths["status"],
            "PASS",
            finalize="PASS",
            compact="PASS",
            policy_split="PASS",
            output_records=str(paths["records"]),
            output_compact=str(paths["compact"]),
            output_policy_split=str(paths["split"]),
            next_step="run three fixed-seed methods after independent artifact audit",
        )
        print(json.dumps(read_json(paths["status"]), indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - preserve a machine-readable failure
        update_status(paths["status"], "FAILED", error=str(exc))
        print(f"aftercare failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
