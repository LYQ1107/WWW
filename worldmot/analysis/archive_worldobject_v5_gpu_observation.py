"""Archive an immutable prefix of the read-only GPU observer with NAS readback."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worldmot.runtime.world_object_v5_contract import v5_source_fingerprint
from worldobject_v5_durable_storage import pinned_nas_root, verified_copy


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    for key in ("repo", "runtime", "monitor-root", "nas-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--launch-receipt", type=Path)
    p.add_argument("--label", required=True)
    args = p.parse_args()
    if not args.label.replace("_", "").isalnum():
        raise ValueError("ALPHANUMERIC_ARCHIVE_LABEL_REQUIRED")
    if args.output.exists():
        raise FileExistsError("PRESERVE_EXISTING_GPU_OBSERVATION_REPORT")
    own_source_sha = digest(Path(__file__))
    source = "c4f9d1c0b2a290062aba52ae7c0d322176ae5376f9c85688f74d89159e16f733"
    if v5_source_fingerprint() != source:
        raise ValueError("FROZEN_MODEL_SOURCE_CHANGED")
    status = json.loads((args.monitor_root / "status.json").read_text())
    protocol_path = args.repo / "worldmot/results/v5_gpu_occupancy_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    monitor = args.repo / "worldmot/analysis/monitor_worldobject_v5_gpu_occupancy.py"
    checker = args.repo / "worldmot/analysis/check_worldobject_v5_gpu_occupancy.py"
    checks = args.repo / "worldmot/results/v5_gpu_occupancy_checks.json"
    launch_path = args.launch_receipt or args.runtime / "gpu_occupancy_monitor_01_launch.json"
    launch = json.loads(launch_path.read_text())
    check_data = json.loads(checks.read_text())
    expected_leaders = {str(x["pid"]): x["start_tick"] for x in protocol["known_goal_leaders"].values()}
    if (status["source_sha256"] != protocol["observer_source_sha256"]
            or digest(monitor) != status["source_sha256"]
            or launch["source_sha256"] != status["source_sha256"]
            or digest(protocol_path) != launch["protocol_sha256"]
            or launch["publication_commit"] != "5143fe83ca65c8be42ce66a645fcef0f88da2bc0"
            or status["known_leaders"] != expected_leaders
            or (status["pid"], status["start_tick"]) != (launch["pid"], launch["start_tick"])
            or check_data["status"] != "PASSED_GENERATED_ONLY_GPU_OBSERVER_CHECKS"
            or check_data["checks"] != 3 or check_data["observer_source_sha256"] != status["source_sha256"]
            or check_data["check_source_sha256"] != digest(checker)):
        raise ValueError("GPU_OBSERVER_SOURCE_OR_PROTOCOL_MISMATCH")
    if status["status"] == "READ_ONLY_GPU_OBSERVER_RUNNING":
        try:
            fields = (Path("/proc") / str(status["pid"]) / "stat").read_text().rsplit(") ", 1)[1].split()
            live = fields[19] == status["start_tick"] and fields[0] != "Z"
        except (OSError, IndexError):
            live = False
        if not live:
            raise ValueError("OBSERVER_RUNNING_STATUS_REQUIRES_LIVE_PID")
    count = status["samples"]
    raw = b"".join((args.monitor_root / "observations.jsonl").read_bytes().splitlines(keepends=True)[:count])
    raw_sha = hashlib.sha256(raw).hexdigest()
    if count <= 0 or raw_sha != status["observations_sha256_at_sample"]:
        raise ValueError("OBSERVATION_PREFIX_SHA_MISMATCH")
    rows = [json.loads(x) for x in raw.splitlines()]
    if (len(rows) != count or sum(x["valid"] for x in rows) != status["valid_samples"]
            or sum(not x["valid"] for x in rows) != status["invalid_samples"]
            or sum(x["goal_contention_observed"] is True for x in rows) != status["samples_with_goal_contention"]
            or rows[-1] != status["last_observation"]
            or any(a["elapsed_seconds"] >= b["elapsed_seconds"] for a, b in zip(rows, rows[1:]))):
        raise ValueError("OBSERVATION_PREFIX_COUNT_OR_ORDER_MISMATCH")
    groups = {}
    for row in rows:
        if not row["valid"]:
            continue
        for device in row["devices"]:
            if not device["goal_processes"]:
                continue
            group = groups.setdefault(device["uuid"], {
                "GPU_UUID": device["uuid"], "index": device["index"], "name": device["name"],
                "samples_with_goal_process": 0, "observed_other_context_samples": 0,
                "samples_with_multiple_goal_processes": 0, "maximum_other_process_count": 0,
                "maximum_other_process_memory_MiB": 0})
            group["samples_with_goal_process"] += 1
            group["observed_other_context_samples"] += int(device["goal_contention_observed"])
            group["samples_with_multiple_goal_processes"] += int(len(device["goal_processes"]) > 1)
            group["maximum_other_process_count"] = max(group["maximum_other_process_count"], device["other_process_count"])
            group["maximum_other_process_memory_MiB"] = max(group["maximum_other_process_memory_MiB"], device["other_process_memory_MiB"])

    prefix = args.runtime / f"GPU_observation_{args.label}_prefix_{count:06d}_{raw_sha[:16]}.jsonl"
    snapshot = args.runtime / f"GPU_observation_{args.label}_status_prefix_{count:06d}.json"
    manifest_path = args.runtime / f"GPU_observation_{args.label}_manifest_prefix_{count:06d}.json"
    temporary = args.runtime / f"GPU_observation_{args.label}_prefix_{count:06d}_build.tar.gz"
    if any(x.exists() for x in (prefix, snapshot, manifest_path, temporary)):
        raise FileExistsError("PRESERVE_EXISTING_GPU_OBSERVATION_SNAPSHOT")
    prefix.write_bytes(raw)
    snapshot.write_text(json.dumps(status, indent=2) + "\n")
    sources = [(prefix, "observations_prefix.jsonl"), (snapshot, "status_at_prefix.json"),
               (launch_path, "launch.json"), (protocol_path, "protocol.json"),
               (monitor, monitor.name), (checker, checker.name), (checks, "generated_checks.json"),
               (Path(__file__), Path(__file__).name)]
    files = [{"path": name, "bytes": path.stat().st_size, "sha256": digest(path)} for path, name in sources]
    manifest_path.write_text(json.dumps(files, indent=2) + "\n")
    with tarfile.open(temporary, "w:gz") as archive:
        for path, name in sources:
            archive.add(path, arcname=name, recursive=False)
        archive.add(manifest_path, arcname="manifest.json", recursive=False)
    with tarfile.open(temporary, "r:gz") as archive:
        if len(archive.getnames()) != len(files) + 1:
            raise ValueError("GPU_ARCHIVE_MEMBER_COUNT_MISMATCH")
        for item in files:
            if hashlib.sha256(archive.extractfile(item["path"]).read()).hexdigest() != item["sha256"]:
                raise ValueError("GPU_ARCHIVE_MEMBER_SHA_MISMATCH")
        if hashlib.sha256(archive.extractfile("manifest.json").read()).hexdigest() != digest(manifest_path):
            raise ValueError("GPU_ARCHIVE_MANIFEST_SHA_MISMATCH")
    for (path, _), item in zip(sources, files):
        if digest(path) != item["sha256"]:
            raise ValueError("GPU_ARCHIVE_SOURCE_CHANGED")
    if digest(Path(__file__)) != own_source_sha:
        raise ValueError("GPU_ARCHIVER_SOURCE_CHANGED")
    archive_sha = digest(temporary)
    name = f"GPU_observation_{args.label}_prefix_{count:06d}_{archive_sha[:16]}.tar.gz"
    local = args.runtime / name
    if local.exists():
        raise FileExistsError("PRESERVE_EXISTING_GPU_OBSERVATION_ARCHIVE")
    temporary.rename(local)
    with pinned_nas_root(args.nas_root) as (anchor, mount):
        readback = verified_copy(local, anchor / "resource_observations" / name, archive_sha)
    receipt = {"verified_UTC": datetime.now(timezone.utc).isoformat(),
               "path": str(args.nas_root / "resource_observations" / name),
               "bytes": local.stat().st_size, "sha256": archive_sha,
               "prefix_observations_sha256": raw_sha, "samples": count,
               "internal_files_verified": len(files) + 1, "mount": mount, "readback": readback}
    result = {"status": "ONGOING_READ_ONLY_GPU_OBSERVATION_VERIFIED_NAS_PREFIX",
              "prefix_started_UTC": rows[0]["UTC"], "prefix_last_UTC": rows[-1]["UTC"],
              "samples": count, "valid_samples": status["valid_samples"],
              "invalid_samples": status["invalid_samples"],
              "samples_with_goal_contention": status["samples_with_goal_contention"],
              "UTC_non_decreasing": all(a["UTC"] <= b["UTC"] for a, b in zip(rows, rows[1:])),
              "observer_source_sha256": status["source_sha256"],
              "archiver_source_sha256": own_source_sha,
              "protocol_publication_commit": launch["publication_commit"],
              "per_goal_GPU": list(groups.values()), "archive": receipt,
              "observer_segment_complete": status["status"] != "READ_ONLY_GPU_OBSERVER_RUNNING",
              "final_complete_observation_log": status["status"] == "OBSERVATION_STOP_FILE_REACHED",
              "continuous_exclusivity_proven": False, "observer_modified_model_inputs_or_parameters": False,
              "observer_discarded_tracking_results": False, "model_forward_calls_for_observer": 0,
              "formal_TEST_or_LIVE_speed_measurement": False}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
