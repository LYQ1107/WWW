"""Dynamic largest-first scheduler for the full formal H=8 replay."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from typing import Any, Mapping, Optional

from run_jev_full_h8_fast_worker import atomic_json, complete_artifact, modify_queue, utc_now
from jev_full_h8_authorization import read_formal_authorization


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
        uuid_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        target_uuid = None
        for line in uuid_result.stdout.splitlines():
            fields = [value.strip() for value in line.split(",", 1)]
            if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) == int(gpu):
                target_uuid = fields[1]
                break
        if target_uuid:
            app_result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,gpu_uuid",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            matched = []
            for line in app_result.stdout.splitlines():
                fields = [value.strip() for value in line.split(",", 1)]
                if len(fields) == 2 and fields[1] == target_uuid and fields[0].isdigit():
                    matched.append(int(fields[0]))
            if matched or int(gpu) != DEFERRED_GPU:
                return matched
    except OSError:
        pass
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
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        for line in result.stdout.splitlines():
            fields = [value.strip() for value in line.split(",", 1)]
            if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) == int(gpu):
                return int(float(fields[1])) * 1024**2
        return 0
    except (OSError, ValueError):
        return 0


def worker_command(args: argparse.Namespace, gpu: int, slot: int = 1) -> list[str]:
    root = Path(__file__).resolve().parents[1]
    worker_id = f"gpu{gpu}" if int(slot) == 1 else f"gpu{gpu}-slot{int(slot)}"
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
        worker_id,
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


def pid_is_worker(pid: Any) -> bool:
    try:
        pid_value = int(pid)
    except (TypeError, ValueError):
        return False
    try:
        command = Path(f"/proc/{pid_value}/cmdline").read_bytes()
    except OSError:
        return False
    command = command.replace(b"\x00", b" ").decode("utf-8", "replace")
    return "run_jev_full_h8_fast_worker.py" in command


def worker_gpu(worker_id: Any) -> Optional[int]:
    match = re.fullmatch(r"gpu(\d+)(?:-slot\d+)?", str(worker_id or ""))
    return int(match.group(1)) if match else None


def recover_stale_queue(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Requeue dead workers and promote already-complete shards.

    The queue is intentionally restart-safe: a supervisor restart must not
    leave an incomplete RUNNING item blocking the full build forever.
    """

    recovered: list[dict[str, Any]] = []

    def update(state: dict[str, Any]):
        for key, item in sorted(state["videos"].items(), key=lambda pair: int(pair[0])):
            video_root = Path(
                item.get("output_root")
                or (args.output_root.resolve() / f"video_{int(key):02d}")
            )
            if complete_artifact(video_root):
                if item.get("status") != "COMPLETE":
                    try:
                        records = int(
                            json.loads(
                                (video_root / "manifest.json").read_text(encoding="utf-8")
                            )["records"]
                        )
                    except (OSError, ValueError, KeyError):
                        records = None
                    item.update(
                        {
                            "status": "COMPLETE",
                            "records": records,
                            "manifest": str(video_root / "manifest.json"),
                            "finished_utc": utc_now(),
                        }
                    )
                    recovered.append(
                        {"video_id": int(key), "event": "promote_complete_shard"}
                    )
                continue
            if item.get("status") != "RUNNING":
                continue
            if pid_is_worker(item.get("pid")):
                continue
            worker = item.get("worker")
            item.update(
                {
                    "status": "PENDING",
                    "worker": None,
                    "pid": None,
                    "records": None,
                    "requeued_reason": "worker PID disappeared; incomplete shard is not official",
                    "requeued_utc": utc_now(),
                }
            )
            recovered.append(
                {
                    "video_id": int(key),
                    "event": "requeue_stale_worker",
                    "worker": worker,
                }
            )

    modify_queue(args.queue, update)
    return recovered


def write_resource_manifest(
    args: argparse.Namespace,
    events: list[dict[str, Any]],
    started: str,
    active_slots: Optional[Mapping[str, Any]] = None,
) -> None:
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
        "slots_per_gpu": int(args.slots_per_gpu),
        "active_slots": dict(active_slots or {}),
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
        "scheduling_policy": "persistent formal GMT workers with bounded slots per safe GPU; largest remaining workload first; no sampling or truncation",
    }
    atomic_json(args.output_root.resolve() / "RESOURCE_SCHEDULING_MANIFEST.json", report)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
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
    parser.add_argument(
        "--formal-gate-report",
        type=Path,
        default=root / "reports/JEV_RNG_V4/FORMAL_GMT_INTRA_VIDEO_CHUNK_EQUIVALENCE_VIDEO07_CURRENT_HEAD_V10.json",
        help="PASS report required before any 24-video canonical H=8 worker is launched",
    )
    parser.add_argument(
        "--slots-per-gpu",
        type=int,
        default=1,
        help="maximum supervised formal replay workers per permitted GPU",
    )
    args = parser.parse_args()
    if any(gpu in FOREIGN_GPUS for gpu in SAFE_GPUS):
        raise AssertionError("safe GPU list overlaps explicitly reserved GPU")
    if args.horizon != 8:
        raise ValueError("full H=8 scheduler is locked to horizon 8")
    if args.slots_per_gpu < 1:
        raise ValueError("slots-per-gpu must be positive")
    formal_gate_report = args.formal_gate_report.resolve()
    read_formal_authorization(formal_gate_report)
    args.output_root.resolve().mkdir(parents=True, exist_ok=True)
    args.log_root.resolve().mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (args.partition_root.resolve() / "partition_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("status") != "COMPLETE":
        raise RuntimeError("partition manifest is not COMPLETE")
    init_queue(args, manifest)
    recovery_events = recover_stale_queue(args)
    started = utc_now()
    events: list[dict[str, Any]] = list(recovery_events)
    children: dict[str, tuple[subprocess.Popen, Any, int]] = {}

    def queue_worker_pids(state: Mapping[str, Any]) -> set[int]:
        return {
            int(item["pid"])
            for item in state["videos"].values()
            if item.get("status") == "RUNNING"
            and pid_is_worker(item.get("pid"))
        }

    def active_slot_counts(state: Mapping[str, Any]) -> dict[int, int]:
        active_workers: set[str] = set()
        for item in state["videos"].values():
            if item.get("status") != "RUNNING" or not pid_is_worker(item.get("pid")):
                continue
            if worker_gpu(item.get("worker")) is not None:
                active_workers.add(str(item.get("worker")))
        # Include children during the short interval before their queue claim
        # becomes visible.  Use worker IDs so a claimed child is not counted
        # twice.
        for worker_id, (process, _handle, _gpu) in children.items():
            if process.poll() is None:
                active_workers.add(worker_id)
        counts: dict[int, int] = {}
        for worker_id in active_workers:
            gpu = worker_gpu(worker_id)
            if gpu is not None:
                counts[gpu] = counts.get(gpu, 0) + 1
        return counts

    def active_worker_ids(state: Mapping[str, Any]) -> set[str]:
        worker_ids = {
            str(item.get("worker"))
            for item in state["videos"].values()
            if item.get("status") == "RUNNING"
            and pid_is_worker(item.get("pid"))
            and worker_gpu(item.get("worker")) is not None
        }
        worker_ids.update(
            worker_id
            for worker_id, (process, _handle, _gpu) in children.items()
            if process.poll() is None
        )
        return worker_ids

    def launch(gpu: int, slot: int, known_pids: set[int]) -> bool:
        worker_id = f"gpu{gpu}" if int(slot) == 1 else f"gpu{gpu}-slot{int(slot)}"
        if worker_id in children:
            return False
        if available_memory() < MIN_AVAILABLE_BYTES:
            events.append(
                {
                    "time": utc_now(),
                    "event": "memory_gate_stop",
                    "gpu": gpu,
                    "slot": slot,
                    "available": available_memory(),
                }
            )
            return False
        pids = gpu_compute_pids(gpu)
        used_bytes = gpu_memory_used_bytes(gpu)
        unknown_pids = sorted(set(pids) - set(known_pids))
        if unknown_pids:
            events.append(
                {
                    "time": utc_now(),
                    "event": "gpu_busy_defer",
                    "gpu": gpu,
                    "slot": slot,
                    "pids": unknown_pids,
                    "memory_used_bytes": used_bytes,
                }
            )
            return False
        # A V100 has 32 GiB.  Normal workers use below 4 GiB; leave driver
        # headroom and refuse to launch into a nearly-full device.
        if used_bytes > 28 * 1024**3:
            events.append(
                {
                    "time": utc_now(),
                    "event": "gpu_memory_gate_stop",
                    "gpu": gpu,
                    "slot": slot,
                    "memory_used_bytes": used_bytes,
                }
            )
            return False
        log_path = args.log_root.resolve() / f"worker_gpu{gpu}_slot{slot}.log"
        handle = log_path.open("a", encoding="utf-8")
        command = worker_command(args, gpu, slot)
        process = subprocess.Popen(
            command,
            cwd=str(root),
            env=worker_environment(gpu),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        children[worker_id] = (process, handle, gpu)
        known_pids.add(process.pid)
        events.append(
            {
                "time": utc_now(),
                "event": "worker_launched",
                "gpu": gpu,
                "slot": slot,
                "worker": worker_id,
                "pid": process.pid,
            }
        )
        return True

    try:
        while True:
            for worker_id, pair in list(children.items()):
                process, handle, gpu = pair
                code = process.poll()
                if code is not None:
                    handle.close()
                    del children[worker_id]
                    events.append(
                        {
                            "time": utc_now(),
                            "event": "worker_exit",
                            "gpu": gpu,
                            "worker": worker_id,
                            "pid": process.pid,
                            "returncode": code,
                        }
                    )

            events.extend(recover_stale_queue(args))
            state = snapshot(args.queue)
            pending = [item for item in state["videos"].values() if item.get("status") == "PENDING"]
            running = [item for item in state["videos"].values() if item.get("status") == "RUNNING"]
            if not pending and not running and not children:
                break

            target_gpus = list(SAFE_GPUS)
            if args.include_gpu0:
                # GPU0 is intentionally polled only after the strict baseline
                # trace process has released it.  No process is terminated here.
                target_gpus.append(DEFERRED_GPU)
            active = active_slot_counts(state)
            active_ids = active_worker_ids(state)
            known_pids = queue_worker_pids(state)
            known_pids.update(
                process.pid
                for process, _handle, _gpu in children.values()
                if process.poll() is None
            )
            for gpu in target_gpus:
                for slot in range(1, int(args.slots_per_gpu) + 1):
                    worker_id = f"gpu{gpu}" if int(slot) == 1 else f"gpu{gpu}-slot{int(slot)}"
                    if worker_id in active_ids:
                        continue
                    if active.get(gpu, 0) >= int(args.slots_per_gpu):
                        break
                    if launch(gpu, slot, known_pids):
                        active[gpu] = active.get(gpu, 0) + 1
                        active_ids.add(worker_id)
            write_resource_manifest(
                args,
                events,
                started,
                active_slots={str(gpu): active.get(gpu, 0) for gpu in target_gpus},
            )
            time.sleep(max(2.0, float(args.poll_seconds)))
    finally:
        for _worker_id, (process, handle, _gpu) in children.items():
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
