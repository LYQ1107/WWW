#!/usr/bin/env python3
"""Dependency scheduler for the already registered six-device experiment.

No model/threshold selection is performed here. Every launched model command goes
through the frozen authorization and real-preflight-gated job launcher. Reports
and scheduling code have separate hashes from the fixed model/input fingerprint.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

FULL="FULL_WORLD_OBJECT_V3"
NG="FULL_NO_GEOMETRY_V3"
HEAD="HEAD_ONLY_NO_GEOMETRY_V3"
NOBIRTH="NO_LEARNED_JOINT_BIRTH_V3"
RULE="RULE_BASED_WORLD_OBJECT_V3"
ROI="LEGACY_PAIR_THEN_MERGE_ROI_RESTORED_CONTROL"
OFF="LEGACY_PAIR_THEN_MERGE_FEATURE_OFF_CONTROL"
GEO="GEOMETRY_ONLY_ABLATION_V3"
TRAINED=(FULL,NG,HEAD)
CONTROLS=(RULE,ROI,OFF,GEO)
METHODS=(FULL,NG,HEAD,NOBIRTH,RULE,ROI,OFF,GEO)
GPUS=("GPU-17231133-4742-e269-cf2b-bfa4bb6e9f8c","GPU-6a5187fc-4b8f-b8cd-ddd1-f76b4be1b68d",
      "GPU-b6f2e85a-b45d-4494-6055-c552327e4f2e","GPU-7c4bd41d-77d6-78e9-e6dc-b1b4b7f473bc",
      "GPU-30aaddaf-5be2-2673-3278-e1e33ce55aa1","GPU-9914f516-bcdd-bef2-d453-bf5e3719a01f")


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp");temp.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n");os.replace(temp,path)
def alive(pid):
    try:os.kill(int(pid),0);return True
    except ProcessLookupError:return False


class Driver:
    def __init__(self,args):
        self.args=args;self.root=args.root;self.repo=args.repo
        self.cache_owner=args.cache_control_root or self.root
        self.preflight_report=args.preflight_report or self.root/"diagnostics/preflight_attempt01/report.json"
        self.source=read(self.preflight_report)["source_sha256"]
        sys.path.insert(0,str(self.repo/"worldmot"))
        from worldmot.runtime.experiment_authorization import source_fingerprint
        if source_fingerprint()!=self.source:raise RuntimeError("MODEL_SOURCE_CHANGED_AFTER_PREFLIGHT")
        self.scheduler_sha=sha(__file__)
        self.env={**os.environ,"PYTHONPATH":str(self.repo/"worldmot"),"CUDA_VISIBLE_DEVICES":"",
            "OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"2",
            "MKL_THREADING_LAYER":"GNU","PYTHONDONTWRITEBYTECODE":"1"}
        self.raw=self.root.parent/"gmt_detector_comparison_20261023"
        self.legacy=self.root.parent/"worldobject_fast_proof_20261025"
        self.original="/data/DATASETS/TRACKING/JDE/VisionTrack"
        self.omega="/data1/liuyeqiang/WWW_omega_runtime_20261016"
        self.actions=[];self.alerts=[]

    def statuses(self):
        values={}
        files=list((self.root/"jobs").glob("*.status.json"))
        if self.cache_owner!=self.root:
            files+=list((self.cache_owner/"jobs").glob("cache_*.status.json"))
        for p in files:
            try:values[p.stem[:-7]]=read(p)
            except (FileNotFoundError,json.JSONDecodeError):continue
        return {v["job_id"]:v for v in values.values()}

    def complete(self,job):return self.statuses().get(job,{}).get("status")=="COMPLETE"

    def occupied(self):
        text=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"],text=True)
        occupied={line.split(",")[0].strip() for line in text.splitlines() if line.strip()}
        for value in self.statuses().values():
            if value.get("status")=="RUNNING" and value.get("gpu_uuid") and alive(value["wrapper_pid"]):occupied.add(value["gpu_uuid"])
        return occupied

    def free_gpu(self,preferred=None):
        occupied=self.occupied();order=[preferred] if preferred else list(GPUS)
        return next((g for g in order if g in GPUS and g not in occupied),None)

    def cpu_available(self):
        return sum(v.get("status")=="RUNNING" and not v.get("gpu_uuid") and alive(v["wrapper_pid"]) for v in self.statuses().values())<3

    def launch(self,name,script,arguments,gpu=None,*,evaluation=False):
        existing=self.statuses().get(name,{})
        if existing:
            if existing.get("status")=="FAILED":self.alerts.append({"job":name,"reason":"ENGINEERING_FAILURE_REQUIRES_INSPECTION"})
            return
        if gpu and gpu in self.occupied():return
        if not gpu and not self.cpu_available():return
        spec={"job_id":name,"script":script,"arguments":[str(x) for x in arguments],"gpu_uuid":gpu,
            "run_root":str(self.root),"source_sha256":self.source,
            "preflight_report":str(self.preflight_report),
            "scheduler_sha256":self.scheduler_sha,"registered_configuration_sha256":sha(self.repo/"worldmot/configs/worldobject_v3_birthfix_registered.json")}
        self.actions.append({"launch":name,"gpu":gpu})
        if self.args.dry_run:return
        path=self.root/"job_specs"/(name+".json")
        if path.exists() and read(path)!=spec:raise RuntimeError("EXISTING_JOB_SPEC_MISMATCH: "+name)
        if not path.exists():atomic(path,spec)
        python=self.args.eval_python if evaluation else self.args.python
        result=subprocess.run([python,str(self.repo/"worldmot/scripts/launch_worldobject_v3_job.py"),"--job",str(path)],env=self.env,cwd=self.repo,text=True,capture_output=True)
        if result.returncode:
            self.alerts.append({"job":name,"reason":result.stdout+result.stderr});return
        print(result.stdout.strip(),flush=True)
        # Wait only for the short wrapper handshake, avoiding duplicate slot use.
        for _ in range(40):
            if name in self.statuses():break
            time.sleep(.1)

    def auth(self,method):return self.root/("authorization_v4.json" if method==GEO else "authorization_v3.json")
    def cache(self,method,split):return self.legacy/f"cache_{split}" if method==NG else self.root/f"cache_{split}_v2"
    def scope(self,method,full=False):
        if method==NG:return self.root/"scopes/FULL_NO_GEOMETRY_V3_v3.json"
        if method in (ROI,OFF):method=FULL
        return self.root/"scopes"/(method+(".full.json" if full else ".train.json"))

    def cache_complete(self,split):
        n,total=(24,28754) if split=="train" else (22,29019)
        manifests=[self.root/f"cache_{split}_v2/scene_{sid:02d}/manifest.json" for sid in range(1,n+1)]
        if not all(p.is_file() for p in manifests):return False
        rows=[read(p) for p in manifests]
        return all(x.get("status")=="COMPLETE" for x in rows) and sum(x["timestamps"] for x in rows)==total

    def freeze(self,method,full=False):
        path=self.scope(method,full)
        if path.is_file():return True
        if self.args.dry_run:self.actions.append({"freeze":str(path)});return False
        args=["--authorization",self.auth(method),"--cache-train",self.cache(method,"train"),
            "--raw-cache-train",self.raw/"raw_train_v1","--method-id",method,"--output",path]
        if full:args.extend(["--cache-test",self.cache(method,"test"),"--raw-cache-test",self.raw/"raw_test_v1"])
        path.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run([self.args.python,str(self.repo/"worldmot/scripts/freeze_worldobject_scope.py"),*map(str,args)],env=self.env,cwd=self.repo,check=True)
        return True

    def gate(self,method):return self.root/"train"/(FULL if method==NOBIRTH else method)/"train_gate.json"

    def train(self,method):
        if not self.freeze(method):return
        slot=GPUS[0] if method==FULL else GPUS[1]
        if self.free_gpu(slot) is None:return
        args=["--authorization",self.auth(method),"--dataset-root",self.original,"--cache",self.cache(method,"train"),
            "--raw-cache",self.raw/"raw_train_v1","--scope-lock",self.scope(method),"--method-id",method,
            "--output",self.root/"train"/method,"--epochs","4","--max-epochs","6","--hidden-dim","128",
            "--seed","20261025","--clip-length","64","--device","cuda:0"]
        self.launch("train_"+method,"train_world_object.py",args,slot)

    def infer(self,method,split):
        name=("infer_" if split=="test" else "train_replay_")+method
        if name in self.statuses():return
        if not self.freeze(FULL if method in (ROI,OFF) else method,full=split=="test"):return
        gpu=None
        if method not in CONTROLS:
            parent=FULL if method==NOBIRTH else method
            if not self.complete("train_"+parent) or not self.gate(method).is_file():return
            gpu=self.free_gpu(GPUS[-1]) if method==NG else self.free_gpu()
            if gpu is None:return
        output=self.root/("predictions" if split=="test" else "train_replay")/method
        if method in (ROI,OFF):
            script="run_world_object_legacy_control.py"
            args=["--authorization",self.auth(method),"--cache",self.cache(method,split),"--raw-cache",self.raw/("raw_"+split+"_v1"),
                "--shared-scope-lock",self.scope(FULL,split=="test"),"--output",output,"--split",split,
                "--feature-mode","restored" if method==ROI else "legacy_disabled"]
        else:
            script="run_world_object_fullvideo.py"
            args=["--authorization",self.auth(method),"--cache",self.cache(method,split),"--raw-cache",self.raw/("raw_"+split+"_v1"),
                "--scope-lock",self.scope(method,split=="test"),"--output",output,"--split",split,"--method-id",method,
                "--device","cuda:0" if gpu else "cpu"]
            args.extend(["--rule-based"] if method in CONTROLS else ["--checkpoint-manifest",self.gate(method)])
        self.launch(name,script,args,gpu)

    def posthoc(self,method):
        if not self.complete("infer_"+method):return
        args=["--authorization",self.auth(method),"--dataset-root",self.original,"--split","test",
            "--audit",self.root.parent/"fulltrain_20261022/audit/data_audit.json","--predictions",self.root/"predictions"/method,
            "--output",self.root/"evaluation"/method,"--method-id",method]
        self.launch("evaluate_"+method,"evaluate_fullvideo.py",args,evaluation=True)
        args=["--authorization",self.auth(method),"--cache",self.cache(method,"test"),"--predictions",self.root/"predictions"/method,
            "--dataset-root",self.original,"--output",self.root/"diagnostics"/("identity_"+method+".json"),
            "--split","test","--method-id",method]
        self.launch("diagnose_"+method,"audit_persistent_world_identity.py",args)

    def live(self,method):
        if not self.complete("infer_"+method):return
        if method in (OFF,GEO):return  # diagnostic controls have no primary live-speed claim
        slot=self.free_gpu(GPUS[-1])
        if slot is None:return
        args=["--authorization",self.auth(method),"--dataset-root",self.root/"metadata_only_dataset",
            "--raw-cache",self.raw/"raw_test_v1","--output",self.root/"live"/method,"--scenes","1","8","22",
            "--split","test","--method-id",method,"--scope-lock",self.scope(FULL if method==ROI else method,True),
            "--device","cuda:0","--max-timestamps","100","--omega-source",self.omega+"/official_vggt_omega",
            "--omega-weights",self.omega+"/weights/vggt_omega_1b_512.pt"]
        args.extend(["--rule-based"] if method in CONTROLS else ["--checkpoint-manifest",self.gate(method)])
        self.launch("live_"+method,"run_world_object_live.py",args,slot)

    def tick(self):
        from worldmot.runtime.experiment_authorization import source_fingerprint
        if source_fingerprint()!=self.source or sha(__file__)!=self.scheduler_sha:raise RuntimeError("SOURCE_CHANGED_STOP_SCHEDULING")
        self.actions=[];self.alerts=[]
        train_done=self.cache_complete("train");test_done=self.cache_complete("test")
        if not train_done and len(list((self.cache_owner/"queue").glob("train_*.pending")))<=2:
            for index in (1,2):
                stop=self.cache_owner/"jobs"/f"cache_worker_{index}.stop"
                if not stop.exists():
                    self.actions.append({"reserve_after_current_scene":index})
                    if not self.args.dry_run:stop.write_text("Finish this scene, then reserve the device for registered supervised training.\n")
        if train_done:
            for method in (FULL,HEAD):self.train(method)
            for method in CONTROLS:self.infer(method,"train")
        for method in METHODS:
            if method==NG or (train_done and test_done):self.infer(method,"test")
            self.posthoc(method)
            self.live(method)
        # Once the early visual experiment and live timing finish, release its
        # sixth slot to the one shared geometry queue, never a duplicate model.
        extra="cache_after_no_geometry"
        if self.complete("live_"+NG) and not test_done and extra not in self.statuses():
            slot=self.free_gpu(GPUS[-1])
            if slot:
                spec=read(self.root/"job_specs/cache_worker_5.json")
                args=spec["arguments"][:];args[args.index("--prefer")+1]="test" if train_done else "train"
                args[args.index("--stop-file")+1]=str(self.root/"jobs"/(extra+".stop"))
                self.launch(extra,"worldobject_v3_cache_worker.py",args,slot)
        status=self.statuses()
        for name,value in status.items():
            if value.get("status")=="FAILED":self.alerts.append({"job":name,"reason":"FAILED_JOB_LOG_PRESERVED"})
        ready=all(self.complete("evaluate_"+m) and self.complete("diagnose_"+m) for m in METHODS)
        ready=ready and all(self.complete("live_"+m) for m in METHODS if m not in (OFF,GEO))
        record={"status":"MEASUREMENTS_READY_FOR_SCIENTIFIC_AUDIT" if ready else "RUNNING_REGISTERED_DEPENDENCIES",
            "updated_at":time.time(),"source_sha256":self.source,"scheduler_sha256":self.scheduler_sha,
            "train_cache_complete":train_done,"test_cache_complete":test_done,"actions":self.actions,
            "alerts":self.alerts,"jobs":{k:{f:v.get(f) for f in ("status","worker_pid","wrapper_pid","gpu_uuid","exit_code")} for k,v in status.items()},
            "goal_complete":False,"remaining_after_measurements":"14-scene error attribution, scientific interpretation, final report and GitHub delivery"}
        if self.args.dry_run:print(json.dumps(record,indent=2));return False
        atomic(self.root/"controller_status.json",record)
        return ready


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",required=True,type=Path);p.add_argument("--repo",required=True,type=Path)
    p.add_argument("--python",default="/data1/liuyeqiang/WWW_omega_runtime_20261016/venv/bin/python")
    p.add_argument("--eval-python",default="/home/liuyeqiang/anaconda3/bin/python")
    p.add_argument("--cache-control-root",type=Path);p.add_argument("--preflight-report",type=Path)
    p.add_argument("--poll-seconds",type=float,default=15);p.add_argument("--once",action="store_true");p.add_argument("--dry-run",action="store_true")
    a=p.parse_args();driver=Driver(a)
    while not (a.root/"controller.stop").exists():
        try:ready=driver.tick()
        except Exception as exc:
            import traceback
            atomic(a.root/"controller_error.json",{"error":str(exc),"traceback":traceback.format_exc(),"time":time.time(),"running_jobs_not_terminated":True});raise
        if ready or a.once or a.dry_run:return
        time.sleep(a.poll_seconds)


if __name__=="__main__":main()
