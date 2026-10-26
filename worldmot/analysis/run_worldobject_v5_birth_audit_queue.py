#!/usr/bin/env python3
"""Run the six preregistered TRAIN attribution scenes, at most two workers."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_worldobject_v5_controller import gpu_snapshot, process_identity, read_json, write_json


def main():
    p = argparse.ArgumentParser()
    for key in ("repo", "runtime", "output-root"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--kind", choices=("birth", "policy"), default="birth")
    a = p.parse_args()
    lock = (a.runtime / "birth_audit_queue.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = a.runtime / "auxiliary_jobs.json"
    jobs = read_json(path, {})
    children = {}
    private = a.output_root / "provenance"
    prefix = "birth_train" if a.kind == "birth" else "policy_train"
    script = "audit_worldobject_v5_birth_train.py" if a.kind == "birth" else "audit_worldobject_v5_frozen_policy_train.py"
    success = "COMPLETE_TRAIN_ONLY_CONDITIONAL_BIRTH_DIAGNOSTIC" if a.kind == "birth" else "COMPLETE_TRAIN_ONLY_FROZEN_POLICY_CLOSED_LOOPS"
    output_prefix = "train_birth_attribution" if a.kind == "birth" else "train_frozen_policy_audit"
    while True:
        for key, job in jobs.items():
            if not key.startswith(prefix + "_") or job["status"] != "RUNNING":
                continue
            if process_identity(job["pid"]) == job["identity"]:
                continue
            proof = Path(job["output"]) / "report.json"
            if proof.exists() and read_json(proof).get("status") == success:
                job["status"] = "COMPLETE"
            else:
                job["status"] = "ENGINEERING_REVIEW_REQUIRED"
            if key in children:
                job["exit_code"] = children[key].poll()
            write_json(path, jobs)
        active = [j for j in jobs.values() if j["status"] == "RUNNING"]
        for scene in (4, 8, 12, 16, 20, 24):
            key = f"{prefix}_scene{scene:02d}"
            if key in jobs or len(active) >= 2:
                continue
            if not os.path.ismount("/data1/liuyeqiang/nas/A100"):
                continue
            main_state = read_json(a.runtime / "controller_state.json", {})
            reserved = {j["GPU_UUID"] for j in main_state.get("jobs", {}).values()
                        if j.get("status") == "RUNNING" and j.get("GPU_UUID")}
            reserved.update(j["GPU_UUID"] for j in active if j.get("GPU_UUID"))
            idle = [g for g in gpu_snapshot() if not g["occupied"] and g["uuid"] not in reserved
                    and "V100" in g["name"] and g["free_MiB"] >= 25000]
            gpu = idle[0] if idle and len(reserved) < 6 else None
            output = a.output_root / f"{output_prefix}/scene_{scene:02d}"
            if output.exists():
                raise FileExistsError("Do not overwrite an unadopted attribution attempt: " + str(output))
            command = [sys.executable, "-u", str(a.repo / "worldmot/analysis" / script),
                       "--scope-lock", str(private / "scope_v5_train.json"),
                       "--authorization", str(private / "authorization_v5_train.json"),
                       "--calibration", str(private / "calibration_train_fit18.json"),
                       "--output", str(output), "--scene", str(scene), "--device", "cuda:0" if gpu else "cpu"]
            env = {**os.environ, "PYTHONPATH": str(a.repo / "worldmot"),
                   "CUDA_VISIBLE_DEVICES": gpu["uuid"] if gpu else "",
                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
            log = a.runtime / f"{prefix}_scene{scene:02d}.log"
            with log.open("x") as stream:
                child = subprocess.Popen(command, cwd=a.repo, env=env, stdout=stream,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            job = {"pid": child.pid, "identity": process_identity(child.pid), "command": command,
                   "status": "RUNNING", "GPU_UUID": gpu["uuid"] if gpu else None,
                   "device": "cuda:0" if gpu else "cpu", "output": str(output), "log": str(log),
                   "started_UTC": datetime.now(timezone.utc).isoformat(), "exit_code": None}
            jobs[key] = job
            children[key] = child
            active.append(job)
            write_json(path, jobs)
            print(json.dumps({"event": "TRAIN_AUDIT_STARTED", "kind": a.kind, "scene": scene,
                              "pid": child.pid, "device": job["device"]}), flush=True)
        own = [j for key, j in jobs.items() if key.startswith(prefix + "_")]
        if len(own) == 6 and all(j["status"] != "RUNNING" for j in own):
            return
        time.sleep(30.)


if __name__ == "__main__":
    main()
