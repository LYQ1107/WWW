"""Dynamic largest-first scheduler for the full formal H=8 replay."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Mapping

from run_jev_full_h8_fast_worker import atomic_json, complete_artifact, modify_queue, utc_now


SAFE_GPUS = (1, 4, 6, 8, 9)
DEFERRED_GPU = 0
FOREIGN_GPUS = (2, 3, 5, 7)
MIN_AVAILABLE_BYTES = 32 * 1024**3


def available_memory() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def gpu_compute_pids(gpu: int) -> list[int]:
    try:
        by_gpu = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu}",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return [int(line.strip()) for line in by_gpu.stdout.splitlines() if line.strip().isdigit()]
    except (OSError, ValueError):
        return []


def gpu_memory_used_bytes(gpu: int) -> int:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        value = next((line.strip() for line in result.stdout.splitlines() if line.strip()), "0")
        return int(float(value)) * 1024**2
    except (OSError, ValueError):
        return 0


def worker_command(args: argparse.Namespace, gpu: int) -> list[str]:
    root = Path(__file__).resolve().parents[1]
    return [
        "/home/liuyeqiang/anaconda3/envs/GMT/bin/python",
        "-u",
        str(root / "reproduction_tools/run_jev_full_h8_fast_worker.py"),
        "--partition-root",
        str(args.partition_root.resolve()),
        "--cache",
        str(args.cache.resolve()),
        "--annotations",
        str(args.annotations.resolve()),
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--config-file",
        str(args.config_file.resolve()),
        "--output-root",
        str(args.output_root.resolve()),
        "--queue",
        str(args.queue.resolve()),
        "--worker-id",
        f"gpu{gpu}",
        "--device",
        "cuda:0",
        "--view-num",
        str(args.view_num),
        "--history-limit",
        str(args.history_limit),
        "--horizon",
        str(args.horizon),
    ]


def worker_environment(gpu: int) -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
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


def init_queue(args: argparse.Namespace, manifest: Mapping[str, Any]) -> None:
    args.queue.parent.mkdir(parents=True, exist_ok=True)
    if args.queue.is_file():
        state = json.loads(args.queue.read_text(encoding="utf-8"))
        if state.get("partition_manifest_sha256") != "sha256:" + __import__("hashlib").sha256(
            (args.partition_root.resolve() / "partition_manifest.json").read_bytes()
        ).hexdigest():
            raise RuntimeError("existing queue belongs to a different partition manifest")
        return
    videos = {}
    for video_id, item in sorted(manifest["videos"].items(), key=lambda pair: int(pair[0])):
        video_root = args.output_root.resolve() / f"video_{int(video_id):02d}"
        status = "COMPLETE" if complete_artifact(video_root) else "PENDING"
        videos[str(video_id)] = {
            "video_id": int(video_id),
            "status": status,
            "worker": None,
            "pid": None,
            "workload_score": int(item["workload_score"]),
            "main_decisions": int(item["main_decisions"]),
            "source_lines": int(item["lines"]),
            "records": None,
            "output_root": str(video_root),
        }
        if status == "COMPLETE":
            try:
                videos[str(video_id)]["records"] = int(
                    json.loads((video_root / "manifest.json").read_text(encoding="utf-8"))["records"]
                )
            except (OSError, ValueError, KeyError):
                videos[str(video_id)]["status"] = "PENDING"
    import hashlib

    partition_hash = hashlib.sha256(
        (args.partition_root.resolve() / "partition_manifest.json").read_bytes()
    ).hexdigest()
    state = {
        "status": "RUNNING",
        "schema_version": 1,
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "partition_root": str(args.partition_root.resolve()),
        "partition_manifest_sha256": "sha256:" + partition_hash,
        "output_root": str(args.output_root.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "horizon": int(args.horizon),
        "association_backend": "formal_gmt_transformer",
        "safe_gpus": list(SAFE_GPUS),
        "deferred_gpu": DEFERRED_GPU,
        "foreign_gpus_never_used": list(FOREIGN_GPUS),
        "videos": videos,
    }
    atomic_json(args.queue, state)


def snapshot(queue_path: Path) -> dict[str, Any]:
    return json.loads(queue_path.read_text(encoding="utf-8"))


def write_resource_manifest(args: argparse.Namespace, events: list[dict[str, Any]], started: str) -> None:
    state = snapshot(args.queue)
    report = {
        "status": "RUNNING" if any(item.get("status") in {"PENDING", "RUNNING"} for item in state["videos"].values()) else "COMPLETE",
        "created_utc": started,
        "updated_utc": utc_now(),
        "partition_root": str(args.partition_root.resolve()),
        "queue": str(args.queue.resolve()),
        "output_root": str(args.output_root.resolve()),
        "safe_gpus": list(SAFE_GPUS),
        "deferred_gpu": DEFERRED_GPU,
        "foreign_gpus_never_used": list(FOREIGN_GPUS),
        "min_mem_available_gate_bytes": MIN_AVAILABLE_BYTES,
        "current_mem_available_bytes": available_memory(),
        "gpu_compute_pids": {str(gpu): gpu_compute_pids(gpu) for gpu in (*SAFE_GPUS, DEFERRED_GPU)},
        "events": events[-200:],
        "video_status": {
            key: {
                "status": value.get("status"),
                "worker": value.get("worker"),
                "records": value.get("records"),
                "workload_score": value.get("workload_score"),
            }
            for key, value in sorted(state["videos"].items(), key=lambda pair: int(pair[0]))
        },
        "scheduling_policy": "one persistent formal GMT worker per safe GPU; largest remaining workload first; no sampling or truncation",
    }
    atomic_json(args.output_root.resolve() / "RESOURCE_SCHEDULING_MANIFEST.json", report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--view-num", type=int, default=2)
    parser.add_argument("--history-limit", type=int, default=80)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--include-gpu0", action="store_true")
    args = parser.parse_args()
    if any(gpu in FOREIGN_GPUS for gpu in SAFE_GPUS):
        raise AssertionError("safe GPU list overlaps explicitly reserved GPU")
    if args.horizon != 8:
        raise ValueError("full H=8 scheduler is locked to horizon 8")
    args.output_root.resolve().mkdir(parents=True, exist_ok=True)
    args.log_root.resolve().mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (args.partition_root.resolve() / "partition_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("status") != "COMPLETE":
        raise RuntimeError("partition manifest is not COMPLETE")
    init_queue(args, manifest)
    started = utc_now()
    events: list[dict[str, Any]] = []
    children: dict[int, subprocess.Popen] = {}
    launched: set[int] = set()

    def launch(gpu: int) -> None:
        if gpu in children:
            return
        if available_memory() < MIN_AVAILABLE_BYTES:
            events.append({"time": utc_now(), "event": "memory_gate_stop", "gpu": gpu, "available": available_memory()})
            return
        pids = gpu_compute_pids(gpu)
        used_bytes = gpu_memory_used_bytes(gpu)
        if pids or used_bytes > 512 * 1024**2:
            events.append({"time": utc_now(), "event": "gpu_busy_defer", "gpu": gpu, "pids": pids, "memory_used_bytes": used_bytes})
            return
        log_path = args.log_root.resolve() / f"worker_gpu{gpu}.log"
        handle = log_path.open("a", encoding="utf-8")
        command = worker_command(args, gpu)
        process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parents[1]),
            env=worker_environment(gpu),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        children[gpu] = (process, handle)
        launched.add(gpu)
        events.append({"time": utc_now(), "event": "worker_launched", "gpu": gpu, "pid": process.pid})

    try:
        while True:
            for gpu, pair in list(children.items()):
                process, handle = pair
                code = process.poll()
                if code is not None:
                    handle.close()
                    del children[gpu]
                    events.append({"time": utc_now(), "event": "worker_exit", "gpu": gpu, "pid": process.pid, "returncode": code})

            state = snapshot(args.queue)
            pending = [item for item in state["videos"].values() if item.get("status") == "PENDING"]
            running = [item for item in state["videos"].values() if item.get("status") == "RUNNING"]
            if not pending and not running and not children:
                break

            for gpu in SAFE_GPUS:
                launch(gpu)
            if args.include_gpu0:
                # GPU0 is intentionally polled only after the strict baseline
                # trace process has released it.  No process is terminated here.
                launch(DEFERRED_GPU)
            write_resource_manifest(args, events, started)
            time.sleep(max(2.0, float(args.poll_seconds)))
    finally:
        for _gpu, (process, handle) in children.items():
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
            handle.close()
    state = snapshot(args.queue)
    failed = [item for item in state["videos"].values() if item.get("status") == "FAILED"]
    state_status = "FAILED" if failed else "COMPLETE"
    modify_queue(args.queue, lambda current: current.update({"status": state_status, "finished_utc": utc_now()}))
    write_resource_manifest(args, events, started)
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
