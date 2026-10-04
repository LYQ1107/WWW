#!/usr/bin/env python3
"""Small dependency-aware scheduler for independent research experiments.

This scheduler is deliberately conservative.  It never kills a process and it
never starts a GPU job on a device whose physical memory is already in use.
GPU numbers are physical numbers, while ``CUDA_VISIBLE_DEVICES`` is set to the
same number for each child.  The canonical GMT Stage2 job is not managed by
this file; it is recorded as an external dependency instead.

Manifest example::

    {"experiments": [{
      "id": "proxy_threshold",
      "gpu": "4",
      "command": ["/path/to/python", "..."],
      "depends_on": ["proxy_events"],
      "success_files": ["outputs/.../metrics.json"],
      "checkpoint": "outputs/stage2_single_gpu/model_4500.pth",
      "checkpoint_kind": "proxy"
    }]}

The scheduler is intended for short, isolated probes.  Final paper numbers
must still be rerun from the canonical Stage2 checkpoint by the main pipeline.
"""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "manifests/parallel_research.json"
STATUS_DIR = ROOT / "outputs/research_parallel"
STATUS_PATH = STATUS_DIR / "status.json"
DASHBOARD_PATH = ROOT / "docs/PARALLEL_EXPERIMENT_STATUS.md"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    experiments = payload.get("experiments")
    if not isinstance(experiments, list):
        raise ValueError("manifest must contain an experiments list")
    seen = set()
    result = []
    for raw in experiments:
        if not isinstance(raw, dict):
            raise ValueError("each experiment must be an object")
        exp = dict(raw)
        exp_id = str(exp.get("id", ""))
        command = exp.get("command")
        if exp.get("external", False):
            command = []
        if not exp_id or exp_id in seen:
            raise ValueError(f"invalid or duplicate experiment id: {exp_id!r}")
        if not isinstance(command, list) or (not command and not exp.get("external", False)):
            raise ValueError(f"{exp_id}: command must be a non-empty list")
        gpu = exp.get("gpu", "")
        if gpu is None:
            gpu = ""
        exp["id"] = exp_id
        exp["gpu"] = str(gpu)
        exp["depends_on"] = [str(item) for item in exp.get("depends_on", [])]
        exp["success_files"] = [str(item) for item in exp.get("success_files", [])]
        exp["command"] = [str(item) for item in command]
        seen.add(exp_id)
        result.append(exp)
    ids = {item["id"] for item in result}
    for exp in result:
        missing = set(exp["depends_on"]) - ids
        if missing:
            raise ValueError(f"{exp['id']}: missing dependencies {sorted(missing)}")
    return result


def gpu_memory() -> Dict[str, int]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
                universal_newlines=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {}
    result: Dict[str, int] = {}
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            result[parts[0]] = int(parts[1])
        except ValueError:
            continue
    return result


def write_state(rows: List[Dict[str, Any]], gpu_state: Optional[Dict[str, int]] = None) -> None:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"updated_utc": now(), "gpu_memory_mib": gpu_state or gpu_memory(), "experiments": rows}
    temporary = STATUS_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(STATUS_PATH)
    lines = [
        "# Parallel Research Status",
        "",
        f"Last update (UTC): `{payload['updated_utc']}`",
        "",
        "GPU0 canonical Stage2 is external to this scheduler and is never stopped or modified.",
        "All proxy rows below are screening-only until rerun from the final canonical checkpoint.",
        "",
        "| Experiment | GPU | Status | PID | Depends on | Checkpoint | Kind | Result |",
        "|---|---:|---|---:|---|---|---|---|",
    ]
    for row in rows:
        deps = ", ".join(row.get("depends_on", [])) or "—"
        result = ", ".join(row.get("success_files", [])) or "—"
        lines.append(
            "| `{id}` | `{gpu}` | **{status}** | `{pid}` | {deps} | `{checkpoint}` | `{kind}` | `{result}` |".format(
                id=row.get("id", ""),
                gpu=row.get("gpu", "") or "CPU",
                status=row.get("status", "PENDING"),
                pid=row.get("pid", "—"),
                deps=deps,
                checkpoint=row.get("checkpoint", "—"),
                kind=row.get("checkpoint_kind", "proxy"),
                result=result,
            )
        )
    lines.extend(
        [
            "",
            "Current physical GPU memory (MiB): "
            + ", ".join(f"GPU{k}={v}" for k, v in sorted((payload["gpu_memory_mib"] or {}).items(), key=lambda item: int(item[0])))
            + ".",
        ]
    )
    DASHBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    DASHBOARD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def success(exp: Dict[str, Any]) -> bool:
    files = exp.get("success_files", [])
    return bool(files) and all(resolve(path).is_file() for path in files)


def start(exp: Dict[str, Any]) -> subprocess.Popen:
    output_dir = exp.get("output_dir")
    if output_dir and resolve(str(output_dir)).exists():
        raise RuntimeError(f"refusing to reuse existing output directory: {output_dir}")
    if output_dir:
        resolve(str(output_dir)).mkdir(parents=True, exist_ok=True)
    log_path = resolve(str(exp.get("log", f"outputs/research_parallel/{exp['id']}.log")))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(
        [
            str(ROOT),
            str(ROOT / "third_party/CenterNet2"),
            str(ROOT / "reproduction_tools"),
            env.get("PYTHONPATH", ""),
        ]
    )
    env["CUDA_VISIBLE_DEVICES"] = str(exp.get("gpu", ""))
    env["OMP_NUM_THREADS"] = str(exp.get("omp_threads", 1))
    if not exp.get("gpu"):
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "1"
    handle = log_path.open("a", encoding="utf-8", buffering=1)
    exp["log"] = str(log_path)
    handle.write(f"[{now()}] $ {shlex.join(exp['command'])}\n")
    handle.flush()
    process = subprocess.Popen(
        exp["command"],
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    exp["_log_handle"] = handle
    return process


def run(manifest_path: Path, poll_seconds: int, once: bool) -> None:
    specs = load_manifest(manifest_path)
    rows = []
    for spec in specs:
        row = dict(spec)
        row.pop("command", None)
        row["status"] = (
            "COMPLETE"
            if success(spec)
            else "WAITING_EXTERNAL"
            if spec.get("external", False)
            else "PENDING"
        )
        row.setdefault("pid", None)
        rows.append(row)
    processes: Dict[str, subprocess.Popen] = {}
    write_state(rows)
    while True:
        memory = gpu_memory()
        reserved_gpus = {gpu for gpu, used in memory.items() if used > 0}
        row_by_id = {row["id"]: row for row in rows}
        for exp in specs:
            row = row_by_id[exp["id"]]
            if exp.get("external", False):
                row["status"] = "COMPLETE" if success(exp) else "WAITING_EXTERNAL"
                continue
            if exp["id"] in processes:
                process = processes[exp["id"]]
                code = process.poll()
                if code is None:
                    continue
                handle = exp.pop("_log_handle", None)
                if handle is not None:
                    handle.close()
                row["return_code"] = code
                row["finished_utc"] = now()
                row["status"] = "COMPLETE" if code == 0 and success(exp) else "FAILED"
                if row["status"] == "FAILED":
                    raise RuntimeError(f"experiment failed: {exp['id']} return_code={code}")
                continue
            if row["status"] == "COMPLETE":
                continue
            dependencies = [row_by_id[item]["status"] for item in exp["depends_on"]]
            if any(status == "FAILED" for status in dependencies):
                row["status"] = "BLOCKED"
                continue
            if any(status != "COMPLETE" for status in dependencies):
                row["status"] = "WAITING_DEPENDENCY"
                continue
            gpu = str(exp.get("gpu", ""))
            if gpu and (memory.get(gpu, 0) > 0 or gpu in reserved_gpus):
                row["status"] = "WAITING_GPU"
                continue
            process = start(exp)
            processes[exp["id"]] = process
            row["status"] = "RUNNING"
            row["pid"] = process.pid
            row["started_utc"] = now()
            if gpu:
                reserved_gpus.add(gpu)
        write_state(rows, memory)
        if once:
            return
        if all(row["status"] == "COMPLETE" for row in rows):
            return
        time.sleep(max(1, poll_seconds))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--once", action="store_true", help="start eligible jobs and exit")
    args = parser.parse_args()
    path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    run(path, args.poll_seconds, args.once)


if __name__ == "__main__":
    main()
