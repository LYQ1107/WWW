#!/usr/bin/env python3
"""Refresh measured tables; run full posthoc attribution only after delivery."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from sample_worldobject_v3_resources import measurements_ready


def atomic(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",required=True,type=Path);p.add_argument("--repo",required=True,type=Path)
    p.add_argument("--python",default="/data1/liuyeqiang/WWW_omega_runtime_20261016/venv/bin/python")
    a=p.parse_args();output=a.repo/"worldmot/results";analysis=a.repo/"worldmot/analysis"
    env={**os.environ,"CUDA_VISIBLE_DEVICES":"","OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1","MKL_THREADING_LAYER":"GNU"}
    previous=None
    while not (a.root/"report_watcher.stop").exists():
        paths=sorted(list((a.root/"train").glob("*/progress.json"))+list((a.root/"train").glob("*/train_gate.json"))
                     +list((a.root/"evaluation").glob("*/metrics.json"))+list((a.root/"live").glob("*/coverage.json")))
        try:signature=[(str(path),path.stat().st_mtime_ns,path.stat().st_size) for path in paths]
        except FileNotFoundError:
            time.sleep(5);continue
        status=a.root/"controller_status.json"
        ready=measurements_ready(status)
        if signature!=previous or ready:
            result=subprocess.run([a.python,str(analysis/"summarize_worldobject_v3.py"),"--run-root",str(a.root),"--output",str(output)],cwd=a.repo,env=env,capture_output=True,text=True)
            if result.returncode:
                atomic(a.root/"report_refresh_error.json",{"status":"RETRY_PENDING","time":time.time(),"exit_code":result.returncode,"stderr":result.stderr[-6000:]})
                time.sleep(60);continue
            print(result.stdout.strip(),flush=True)
            atomic(a.root/"publication_required.json",{"status":"NEW_MEASURED_ARTIFACT_SNAPSHOT","updated_at":time.time(),
                "artifacts":signature,"scientific_goal_complete":False,"GitHub_push_still_required":True})
            previous=signature
        if ready:
            subprocess.run([a.python,str(analysis/"audit_worldobject_v3_identities.py"),"--run-root",str(a.root),"--output",str(output)],cwd=a.repo,env=env,check=True)
            atomic(a.root/"final_analysis_ready.json",{"status":"FULL_POSTHOC_TABLES_READY_FOR_REVIEW","finished_at":time.time(),
                "analysis_source_sha256":hashlib.sha256((analysis/"audit_worldobject_v3_identities.py").read_bytes()).hexdigest(),
                "scientific_interpretation_and_GitHub_delivery_remaining":True,"goal_complete":False})
            return
        time.sleep(60)


if __name__=="__main__":main()
