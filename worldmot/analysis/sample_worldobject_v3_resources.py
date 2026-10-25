#!/usr/bin/env python3
"""NVML observation only: creates no CUDA context and owns no model process."""
from __future__ import annotations
import argparse
import csv
import ctypes as C
import json
from pathlib import Path
import subprocess
import time

from drive_worldobject_v3 import GPUS

FIELDS=["time_unix","gpu_uuid","device_used_bytes","device_total_bytes","gpu_utilization_pct","jobs","nvml_status"]


def measurements_ready(path):
    try:return json.loads(Path(path).read_text()).get("status")=="MEASUREMENTS_READY_FOR_SCIENTIFIC_AUDIT"
    except (FileNotFoundError,json.JSONDecodeError):return False


def resource_log_mode(path,resume):
    path=Path(path)
    if not path.exists():return "x",True
    if not resume:raise RuntimeError("PRESERVE_EXISTING_RESOURCE_LOG_BEFORE_RESTART")
    with path.open() as stream:
        if next(csv.reader(stream),None)!=FIELDS:raise RuntimeError("RESOURCE_LOG_HEADER_MISMATCH")
    with path.open("rb") as stream:
        stream.seek(-1,2)
        if stream.read(1)!=b"\n":raise RuntimeError("PRESERVE_INCOMPLETE_RESOURCE_LOG_BEFORE_RESUME")
    return "a",False


class Memory(C.Structure):
    _fields_=[("total",C.c_ulonglong),("free",C.c_ulonglong),("used",C.c_ulonglong)]


class Utilization(C.Structure):
    _fields_=[("gpu",C.c_uint),("memory",C.c_uint)]


def read_jobs(root):
    result={}
    for p in (root/"jobs").glob("*.status.json"):
        try:d=json.loads(p.read_text())
        except (FileNotFoundError,json.JSONDecodeError):continue
        if d.get("status")=="RUNNING" and d.get("gpu_uuid") in GPUS:
            result.setdefault(d["gpu_uuid"],[]).append(d["job_id"])
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",required=True,type=Path);p.add_argument("--interval",type=float,default=.2)
    p.add_argument("--resume",action="store_true")
    a=p.parse_args();lib=C.CDLL("libnvidia-ml.so.1")
    if lib.nvmlInit_v2()!=0:raise RuntimeError("NVML_INIT_FAILED")
    handles={}
    for uuid in GPUS:
        handle=C.c_void_p()
        if lib.nvmlDeviceGetHandleByUUID(uuid.encode(),C.byref(handle))!=0:raise RuntimeError("NVML_UUID_LOOKUP_FAILED")
        handles[uuid]=handle
    jobs={};refresh=0.;flush=0.;started=time.time()
    path=a.root/"logs/resource_samples.csv"
    mode,header=resource_log_mode(path,a.resume)
    with path.open(mode) as stream,(a.root/"logs/resource_process_samples.jsonl").open("a" if a.resume else "x") as processes:
        writer=csv.DictWriter(stream,fieldnames=FIELDS)
        if header:writer.writeheader()
        processes.write(json.dumps({"event":"SAMPLER_RESUME" if a.resume else "SAMPLER_START","time_unix":started})+"\n")
        while not (a.root/"resource_sampler.stop").exists():
            now=time.time()
            if now-refresh>=5:
                jobs=read_jobs(a.root);refresh=now
                try:
                    text=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,used_memory","--format=csv,noheader"],text=True,timeout=3)
                    rows=[line for line in text.splitlines() if line.split(",")[0].strip() in GPUS]
                    processes.write(json.dumps({"time_unix":time.time(),"compute_processes":rows,"jobs":jobs})+"\n")
                except subprocess.TimeoutExpired:
                    processes.write(json.dumps({"time_unix":time.time(),"status":"PROCESS_QUERY_TIMEOUT_NVML_MEMORY_SAMPLING_CONTINUES"})+"\n")
                except subprocess.CalledProcessError as exc:
                    processes.write(json.dumps({"time_unix":time.time(),"status":"PROCESS_QUERY_FAILED_MEMORY_SAMPLING_CONTINUES","exit_code":exc.returncode})+"\n")
                if measurements_ready(a.root/"controller_status.json"):break
            for uuid,handle in handles.items():
                memory=Memory();util=Utilization()
                rc=lib.nvmlDeviceGetMemoryInfo(handle,C.byref(memory));urc=lib.nvmlDeviceGetUtilizationRates(handle,C.byref(util))
                writer.writerow({"time_unix":time.time(),"gpu_uuid":uuid,"device_used_bytes":memory.used if rc==0 else None,
                    "device_total_bytes":memory.total if rc==0 else None,"gpu_utilization_pct":util.gpu if urc==0 else None,
                    "jobs":"|".join(jobs.get(uuid,[])),"nvml_status":rc})
            if now-flush>=5:stream.flush();processes.flush();flush=now
            time.sleep(a.interval)
    lib.nvmlShutdown()
    (a.root/"resource_sampler_complete.json").write_text(json.dumps({"started_at":started,"finished_at":time.time(),
        "sampling_interval_s":a.interval,"peak_definition":"maximum observed total device allocation, not an exact allocation-event trace",
        "includes_GMT_child_context":True,"other_users_devices_sampled":False},indent=2))


if __name__=="__main__":main()
