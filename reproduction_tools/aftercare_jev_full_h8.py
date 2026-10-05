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
        "methods": runtime / "FIRST_ROUND_METHODS",
        "first_round_report": runtime / "FIRST_ROUND_REPORT.json",
        "first_round_markdown": runtime / "FIRST_ROUND_REPORT.md",
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


def environment(root: Path, cuda_visible: str = "") -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible
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


def run_step(
    name: str,
    command: list[Path | str],
    log: Path,
    root: Path,
    *,
    cuda_visible: str = "",
) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc_now()}] {name}: $ {' '.join(str(item) for item in command)}\n")
        handle.flush()
        result = subprocess.run(
            [str(item) for item in command],
            cwd=root,
            env=environment(root, cuda_visible),
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


def capacity_match(root: Path, state_dim: int) -> dict[str, Any]:
    """Select widths by the same <2% trainable-parameter gate as aggregation."""
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "reproduction_tools"))
    from train_jev import choose_model

    target_name = "jev"
    target_width = 64

    def count(name: str, width: int) -> int:
        model = choose_model(name, state_dim, width)
        return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))

    target_params = count(target_name, target_width)
    result: dict[str, Any] = {
        "target_model": target_name,
        "target_hidden_dim": target_width,
        "target_trainable_params": target_params,
        "state_dim": int(state_dim),
        "methods": {},
    }
    for name in ("question_threshold", "question_conditioned_mlp"):
        candidates = []
        for width in range(8, 129):
            params = count(name, width)
            relative = abs(params - target_params) / max(1, target_params)
            candidates.append((relative, width, params))
        relative, width, params = min(candidates)
        if relative >= 0.02:
            raise RuntimeError(f"capacity matching failed for {name}: relative difference={relative}")
        result["methods"][name] = {
            "hidden_dim": int(width),
            "trainable_params": int(params),
            "relative_difference": float(relative),
        }
    result["methods"][target_name] = {
        "hidden_dim": target_width,
        "trainable_params": target_params,
        "relative_difference": 0.0,
    }
    return result


def live_formal_workers() -> list[int]:
    workers = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if "run_jev_full_h8_fast_worker.py" in command:
            workers.append(int(entry.name))
    return sorted(workers)


def gpu_compute_pids(gpu: int) -> list[int]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={int(gpu)}",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return []
    return [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]


def wait_for_training_gpus(
    status_path: Path,
    *,
    gpus: tuple[int, ...],
    poll_seconds: float,
) -> None:
    while True:
        formal = live_formal_workers()
        busy = {str(gpu): gpu_compute_pids(gpu) for gpu in gpus}
        if not formal and not any(busy.values()):
            update_status(status_path, "RUNNING", training_gpu_gate="PASS", training_gpus=list(gpus))
            return
        update_status(
            status_path,
            "RUNNING",
            training_gpu_gate="WAITING",
            formal_worker_pids=formal,
            training_gpu_compute_pids=busy,
        )
        time.sleep(max(5.0, float(poll_seconds)))


def validate_existing(paths: Mapping[str, Path], expected_records: int, seed: int) -> dict[str, bool]:
    ready = {
        "finalize": False,
        "compact": False,
        "split": False,
        "first_round": False,
    }
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
        if int(payload.get("seed", -1)) != seed or payload.get("official_test_used_for_search") is not False:
            raise RuntimeError("existing policy split is not the locked TRAIN-only seed split")
        ready["split"] = True

    report = paths["first_round_report"]
    markdown = paths["first_round_markdown"]
    if report.exists() or markdown.exists():
        if not report.is_file() or not markdown.is_file():
            raise RuntimeError("partial first-round report exists; refusing overwrite")
        payload = read_json(report)
        if payload.get("status") != "PASS" or list(payload.get("seeds", ())) != [seed]:
            raise RuntimeError("existing first-round report is not the locked single-seed result")
        ready["first_round"] = True
    return ready


def run_first_round(
    paths: Mapping[str, Path],
    root: Path,
    *,
    seed: int,
    poll_seconds: float,
    log: Path,
) -> dict[str, Any]:
    compact_manifest = read_json(paths["compact"] / "manifest.json")
    state_dim = int(compact_manifest["state_dim"])
    capacity = capacity_match(root, state_dim)
    method_specs = (
        ("question_threshold", 4),
        ("question_conditioned_mlp", 6),
        ("jev", 8),
    )
    wait_for_training_gpus(paths["status"], gpus=tuple(gpu for _, gpu in method_specs), poll_seconds=poll_seconds)
    paths["methods"].mkdir(parents=True, exist_ok=True)
    processes: list[tuple[str, subprocess.Popen, Any, int]] = []
    for method, gpu in method_specs:
        hidden_dim = int(
            capacity["methods"][method]["hidden_dim"]
        )
        output = paths["methods"] / method
        method_log = paths["methods"] / f"{method}.launcher.log"
        command = [
            PYTHON,
            "-u",
            root / "reproduction_tools/run_jev_three_way_method.py",
            "--dataset",
            paths["compact"],
            "--split-manifest",
            paths["split"],
            "--output",
            output,
            "--model",
            method,
            "--hidden-dim",
            str(hidden_dim),
            "--epochs",
            "50",
            "--batch-size",
            "128",
            "--lr",
            "1e-3",
            "--device",
            "cuda:0",
        ]
        method_log.parent.mkdir(parents=True, exist_ok=True)
        handle = method_log.open("a", encoding="utf-8")
        handle.write(f"[{utc_now()}] gpu={gpu} $ {' '.join(str(item) for item in command)}\n")
        handle.flush()
        process = subprocess.Popen(
            [str(item) for item in command],
            cwd=root,
            env=environment(root, str(gpu)),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append((method, process, handle, gpu))
    update_status(
        paths["status"],
        "RUNNING",
        first_round_training="RUNNING",
        seed=seed,
        capacity=capacity,
        method_gpus={method: gpu for method, gpu in method_specs},
    )
    failed: list[tuple[str, int | None]] = []
    while processes:
        remaining: list[tuple[str, subprocess.Popen, Any, int]] = []
        for method, process, handle, gpu in processes:
            code = process.poll()
            if code is None:
                remaining.append((method, process, handle, gpu))
            else:
                handle.close()
                if code:
                    failed.append((method, code))
        processes = remaining
        if processes:
            time.sleep(min(30.0, max(5.0, float(poll_seconds))))
    if failed:
        raise RuntimeError(f"first-round method training failed: {failed}")

    aggregation_command = [
        PYTHON,
        "-u",
        root / "reproduction_tools/aggregate_jev_three_way_compact.py",
        "--dataset",
        paths["compact"],
        "--split-manifest",
        paths["split"],
        "--methods-root",
        paths["methods"],
        "--output",
        paths["first_round_report"],
        "--markdown",
        paths["first_round_markdown"],
    ]
    run_step("first_round_aggregate", aggregation_command, log, root)
    report = read_json(paths["first_round_report"])
    if report.get("status") != "PASS" or list(report.get("seeds", ())) != [seed]:
        raise RuntimeError("first-round aggregation did not produce a locked single-seed PASS")
    return {"capacity": capacity, "gate": report.get("jev_beats_both_on_val_utility"), "report": str(paths["first_round_report"])}


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

        existing = validate_existing(paths, expected_records, SEED)
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
        update_status(paths["status"], "RUNNING", finalize="PASS", compact="PASS", policy_split="PASS")
        first_round: Mapping[str, Any]
        if existing["first_round"]:
            first_round = {
                "report": str(paths["first_round_report"]),
                "gate": read_json(paths["first_round_report"]).get("jev_beats_both_on_val_utility"),
            }
        else:
            first_round = run_first_round(
                paths,
                root,
                seed=SEED,
                poll_seconds=float(args.poll_seconds),
                log=paths["log"],
            )
        update_status(
            paths["status"],
            "PASS",
            finalize="PASS",
            compact="PASS",
            policy_split="PASS",
            first_round="PASS",
            first_round_report=str(paths["first_round_report"]),
            jev_beats_both_on_val_utility=first_round.get("gate"),
            output_records=str(paths["records"]),
            output_compact=str(paths["compact"]),
            output_policy_split=str(paths["split"]),
            next_step="run formal tracking comparison only if the single-seed JEV utility gate is true",
        )
        print(json.dumps(read_json(paths["status"]), indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - preserve a machine-readable failure
        update_status(paths["status"], "FAILED", error=str(exc))
        print(f"aftercare failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
