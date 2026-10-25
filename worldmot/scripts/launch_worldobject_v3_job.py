#!/usr/bin/env python3
"""Audited single-job launcher; the agent schedules completed slots dynamically."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from worldmot.runtime.experiment_authorization import source_commit,source_fingerprint

ALLOWED={"worldobject_v3_cache_worker.py","train_world_object.py","run_world_object_fullvideo.py",
         "run_world_object_legacy_control.py","run_world_object_live.py","evaluate_fullvideo.py",
         "audit_persistent_world_identity.py","audit_world_object_support.py","evaluate_world_object_formation.py"}


def alive(pid):
    try:os.kill(int(pid),0);return True
    except ProcessLookupError:return False


def main():
    p=argparse.ArgumentParser();p.add_argument("--job",required=True,type=Path);p.add_argument("--run-child",action="store_true")
    a=p.parse_args();job=json.loads(a.job.read_text());root=Path(job["run_root"])
    preflight=json.loads(Path(job["preflight_report"]).read_text())
    if preflight.get("status")!="REAL_PREFLIGHT_PASS" or job["source_sha256"]!=source_fingerprint():raise RuntimeError("PREFLIGHT_OR_SOURCE_NOT_READY")
    if preflight.get("source_sha256")!=job["source_sha256"]:raise RuntimeError("SOURCE_CHANGED_SINCE_REAL_PREFLIGHT")
    if job["script"] not in ALLOWED:raise ValueError("UNREGISTERED_JOB_SCRIPT")
    metadata=root/"jobs"/(job["job_id"]+".status.json");metadata.parent.mkdir(parents=True,exist_ok=True)
    env={**os.environ,"PYTHONPATH":str(Path(__file__).resolve().parents[1]),"OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2",
         "OPENBLAS_NUM_THREADS":"1","MKL_THREADING_LAYER":"GNU","PYTHONDONTWRITEBYTECODE":"1",
         "CUDA_VISIBLE_DEVICES":job.get("gpu_uuid") or ""}
    if not a.run_child:
        if metadata.is_file():
            old=json.loads(metadata.read_text())
            if old.get("status")=="COMPLETE" or (old.get("status")=="RUNNING" and alive(old["wrapper_pid"])):
                print(json.dumps(old));return
        if job.get("gpu_uuid"):
            apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"],text=True)
            if any(line.split(',')[0].strip()==job["gpu_uuid"] for line in apps.splitlines()):raise RuntimeError("SELECTED_GPU_NOW_OCCUPIED_DO_NOT_DISTURB")
            running=[json.loads(path.read_text()) for path in metadata.parent.glob("*.status.json")]
            if sum(x.get("status")=="RUNNING" and x.get("gpu_uuid") and alive(x["wrapper_pid"]) for x in running)>=6:
                raise RuntimeError("SIX_GPU_CONCURRENCY_LIMIT")
        log=root/"logs"/(job["job_id"]+".log");log.parent.mkdir(parents=True,exist_ok=True)
        with log.open("a") as stream:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--job",str(a.job.resolve()),"--run-child"],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({"job_id":job["job_id"],"wrapper_pid":process.pid,"log":str(log)}));return
    record={"job_id":job["job_id"],"status":"RUNNING","wrapper_pid":os.getpid(),"gpu_uuid":job.get("gpu_uuid"),
            "source_commit":source_commit(),"source_sha256":source_fingerprint(),"started_at":time.time(),"command":[job["script"],*job["arguments"]]}
    metadata.write_text(json.dumps(record,indent=2))
    process=subprocess.Popen([sys.executable,str(Path(__file__).with_name(job["script"])),*job["arguments"]],env=env)
    record["worker_pid"]=process.pid;metadata.write_text(json.dumps(record,indent=2))
    code=process.wait();record.update(status="COMPLETE" if code==0 else "FAILED",exit_code=code,finished_at=time.time())
    metadata.write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)


if __name__=="__main__":main()
