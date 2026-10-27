"""Read-only GPU occupancy evidence; never controls models or other processes."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


def utc():
    return datetime.now(timezone.utc).isoformat()


def process_identity(pid, proc=Path("/proc")):
    try:
        fields = (proc / str(pid) / "stat").read_text().rsplit(") ", 1)[1].split()
        return int(fields[1]), fields[19]
    except (OSError, ValueError, IndexError):
        return None


def belongs_to_goal(pid, leaders, proc=Path("/proc")):
    visited = set()
    for _ in range(32):
        if pid <= 1 or pid in visited:
            return False
        visited.add(pid)
        identity = process_identity(pid, proc)
        if identity is None:
            return False
        parent, start_tick = identity
        if pid in leaders:
            return start_tick == leaders[pid]
        pid = parent
    return False


def rows(text, width):
    parsed = [[x.strip() for x in row] for row in csv.reader(text.splitlines()) if row]
    if any(len(row) != width or any(not x for x in row) for row in parsed):
        raise ValueError("MALFORMED_NVIDIA_QUERY")
    return parsed


def occupancy(devices_text, apps_text, leaders, proc=Path("/proc")):
    devices = {}
    for index, uuid, name in rows(devices_text, 3):
        if uuid in devices:
            raise ValueError("DUPLICATE_GPU_UUID")
        devices[uuid] = {"index": int(index), "uuid": uuid, "name": name,
                         "goal_processes": [], "other_process_count": 0,
                         "other_process_memory_MiB": 0}
    if not devices:
        raise ValueError("NO_GPU_DEVICES_OBSERVED")
    self_on_GPU = False
    for pid_text, uuid, memory_text in rows(apps_text, 3):
        pid, memory = int(pid_text), int(memory_text)
        if uuid not in devices or pid <= 0 or memory < 0:
            raise ValueError("INVALID_GPU_APPLICATION_ROW")
        self_on_GPU |= pid == os.getpid()
        if belongs_to_goal(pid, leaders, proc):
            identity = process_identity(pid, proc)
            devices[uuid]["goal_processes"].append({
                "pid": pid, "start_tick": identity[1] if identity else None,
                "memory_MiB": memory})
        else:
            # No user names, commands, paths or unrelated PIDs enter this log.
            devices[uuid]["other_process_count"] += 1
            devices[uuid]["other_process_memory_MiB"] += memory
    if self_on_GPU:
        raise ValueError("READ_ONLY_MONITOR_UNEXPECTEDLY_ON_GPU")
    for device in devices.values():
        device["compute_idle_at_sample"] = not device["goal_processes"] and not device["other_process_count"]
        device["goal_contention_observed"] = bool(device["goal_processes"] and device["other_process_count"])
    return list(devices.values())


def nvidia_query(columns):
    return subprocess.run(["nvidia-smi", columns, "--format=csv,noheader,nounits"],
                          check=True, capture_output=True, text=True, timeout=15).stdout


def sample(leaders, runner=nvidia_query, proc=Path("/proc")):
    result = {"UTC": utc(), "valid": False}
    try:
        result["devices"] = occupancy(
            runner("--query-gpu=index,uuid,name"),
            runner("--query-compute-apps=pid,gpu_uuid,used_memory"), leaders, proc)
        result["valid"] = True
        result["goal_contention_observed"] = any(d["goal_contention_observed"] for d in result["devices"])
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)[:500]
        # Missing evidence is not an idle GPU observation.
        result["goal_contention_observed"] = None
    return result


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".partial." + str(os.getpid()))
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--known-leader", action="append", default=[], help="PID:START_TICK")
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--interval", type=float, default=5.)
    parser.add_argument("--max-seconds", type=float, default=28800.)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if (source_sha != args.expected_source_sha or not math.isfinite(args.interval)
            or not math.isfinite(args.max_seconds) or not 1 <= args.interval <= 60 or args.max_seconds <= 0):
        raise ValueError("FROZEN_OBSERVER_SOURCE_AND_BOUNDED_INTERVAL_REQUIRED")
    leaders = {int(x.split(":", 1)[0]): x.split(":", 1)[1] for x in args.known_leader}
    if not leaders:
        raise ValueError("KNOWN_GOAL_PROCESS_IDENTITIES_REQUIRED")
    if args.once:
        observation = sample(leaders)
        print(json.dumps(observation))
        return 0 if observation["valid"] else 2
    if args.output_root is None:
        raise ValueError("OUTPUT_ROOT_REQUIRED")
    args.output_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    state = {"status": "READ_ONLY_GPU_OBSERVER_RUNNING", "pid": os.getpid(),
             "start_tick": process_identity(os.getpid())[1], "started_UTC": utc(),
             "source_sha256": source_sha, "known_leaders": leaders,
             "interval_seconds": args.interval, "maximum_seconds": args.max_seconds,
             "samples": 0, "valid_samples": 0, "invalid_samples": 0,
             "samples_with_goal_contention": 0, "model_processes_modified": False,
             "GT_read": False, "GPU_allocation_requested": False,
             "sampling_does_not_prove_continuous_exclusivity": True}
    digest = hashlib.sha256()
    status_path = args.output_root / "status.json"
    # Exclusive creation prevents a second observer from replacing prior evidence.
    with (args.output_root / "observations.jsonl").open("xb") as stream:
        while True:
            tick = time.monotonic()
            observation = sample(leaders)
            observation["elapsed_seconds"] = time.monotonic() - started
            line = (json.dumps(observation, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
            digest.update(line)
            state["samples"] += 1
            state["valid_samples"] += int(observation["valid"])
            state["invalid_samples"] += int(not observation["valid"])
            state["samples_with_goal_contention"] += int(observation["goal_contention_observed"] is True)
            state["last_UTC"] = observation["UTC"]
            state["last_observation"] = observation
            state["observations_sha256_at_sample"] = digest.hexdigest()
            stop = args.stop_file is not None and args.stop_file.is_file()
            limit = time.monotonic() - started >= args.max_seconds
            if stop or limit:
                state["status"] = "OBSERVATION_STOP_FILE_REACHED" if stop else "BOUNDED_OBSERVATION_COMPLETE"
            atomic_json(status_path, state)
            if stop or limit:
                break
            time.sleep(max(0., args.interval - (time.monotonic() - tick)))
    print(json.dumps({k: v for k, v in state.items() if k != "last_observation"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
