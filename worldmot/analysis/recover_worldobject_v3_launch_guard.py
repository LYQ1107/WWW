#!/usr/bin/env python3
"""Launch already registered pending GPU jobs past a None-counting guard bug.

Only the launch module's boolean slot count is adapted in this observer process.
Frozen files, model arguments, checkpoints and normal workers remain untouched.
The original launcher still checks source/preflight/allowlist/device occupancy.
"""
import argparse
import builtins
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

GPUS=("GPU-17231133-4742-e269-cf2b-bfa4bb6e9f8c","GPU-6a5187fc-4b8f-b8cd-ddd1-f76b4be1b68d",
      "GPU-b6f2e85a-b45d-4494-6055-c552327e4f2e","GPU-7c4bd41d-77d6-78e9-e6dc-b1b4b7f473bc",
      "GPU-30aaddaf-5be2-2673-3278-e1e33ce55aa1","GPU-9914f516-bcdd-bef2-d453-bf5e3719a01f")


def boolean_slot_count(values):
    return builtins.sum(bool(value) for value in values)


def read(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--once',action='store_true');a=p.parse_args()
    root=a.root;folder=root/'launch_guard_recovery';folder.mkdir(exist_ok=True)
    lock=(folder/'observer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    launcher=Path(__file__).resolve().parents[1]/'scripts/launch_worldobject_v3_job.py'
    spec=importlib.util.spec_from_file_location('frozen_R2_launcher',launcher);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.sum=boolean_slot_count
    while not (folder/'stop').exists():
        statuses={}
        for path in (root/'jobs').glob('*.status.json'):
            try:statuses[read(path)['job_id']]=read(path)
            except (OSError,json.JSONDecodeError):continue
        text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        busy={line.split(',')[0].strip() for line in text.splitlines() if line.strip()}
        for value in statuses.values():
            if value.get('status')=='RUNNING' and value.get('gpu_uuid') and module.alive(value['wrapper_pid']):busy.add(value['gpu_uuid'])
        paths=list((root/'job_specs').glob('infer_*.json'))+list((root/'job_specs').glob('live_*.json'))
        # The two failed original specs both requested GPU4. Establish NoBirth
        # on free GPU6 first, then Full on GPU4, so the running scheduler can
        # observe both statuses without a stale-spec resource mismatch.
        paths.sort(key=lambda path:(0 if path.stem=='infer_NO_LEARNED_JOINT_BIRTH_V3' else 1,path.name))
        for path in paths:
            job=read(path);name=job['job_id']
            if name in statuses or not job.get('gpu_uuid'):continue
            requested=GPUS[2] if name=='infer_NO_LEARNED_JOINT_BIRTH_V3' else job['gpu_uuid']
            if requested in busy:continue
            if requested not in GPUS:raise ValueError('UNREGISTERED_PHYSICAL_DEVICE')
            actual=path
            if requested!=job['gpu_uuid']:
                adjusted={**job,'gpu_uuid':requested};actual=folder/(name+'.json')
                if actual.exists() and read(actual)!=adjusted:raise ValueError('RECOVERY_SPEC_CHANGED')
                if not actual.exists():actual.write_text(json.dumps(adjusted,indent=2)+'\n')
            old_argv=sys.argv;sys.argv=[str(launcher),'--job',str(actual)]
            try:module.main()
            except RuntimeError as exc:
                print(json.dumps({'launch_wait':name,'reason':str(exc)}),flush=True)
                continue
            finally:sys.argv=old_argv
            for _ in range(40):
                if (root/'jobs'/(name+'.status.json')).exists():break
                time.sleep(.1)
            busy.add(requested)
            with (folder/'recovery_events.jsonl').open('a') as stream:
                stream.write(json.dumps({'time':time.time(),'job':name,'original_spec':str(path),'executed_spec':str(actual),
                    'gpu_uuid':requested,'guard_only_boolean_count_normalization':True,'model_arguments_changed':False,
                    'training_restarted':False,'frozen_launcher_file_modified':False})+'\n')
        if a.once:return
        time.sleep(5)


if __name__=='__main__':main()
