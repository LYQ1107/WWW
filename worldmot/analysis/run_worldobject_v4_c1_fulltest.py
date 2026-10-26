#!/usr/bin/env python3
"""Scene scheduler for the one authorized C1 TEST; no metric-based decisions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from worldmot.runtime.world_object_v4_test_contract import (require_scope,completed_scene,
    write_json,source_fingerprint,sha256)


def now():return datetime.now(timezone.utc).isoformat()


def idle_gpus():
    result=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True,stdout=subprocess.PIPE,check=True,timeout=15)
    rows=[]
    for line in result.stdout.splitlines():
        index,uuid,memory,util=[x.strip() for x in line.split(',')]
        if 4<=int(index)<=9 and int(memory)<256 and int(util)==0:rows.append((int(index),uuid))
    return rows


def archive_scene(local,nas,sid):
    source=local/f'predictions/V4_C1/scene_{sid:02d}'
    manifest=json.loads((source/'manifest.json').read_text());target=nas/f'predictions/V4_C1/scene_{sid:02d}'
    expected={**manifest['output_sha256'],'manifest.json':sha256(source/'manifest.json')}
    if target.exists():
        if all(sha256(target/n)==v for n,v in expected.items()):return {'scene':sid,'status':'VERIFIED_ALREADY_ARCHIVED'}
        raise ValueError('NEVER_OVERWRITE_DIFFERENT_ARCHIVED_SCENE '+str(sid))
    stage=target.with_name('.'+target.name+'.staging.'+str(os.getpid()));stage.mkdir(parents=True,exist_ok=False)
    for name,digest in expected.items():
        shutil.copyfile(source/name,stage/name)
        if sha256(stage/name)!=digest:raise ValueError('NAS_COPY_SHA_MISMATCH '+name)
    if target.exists():raise FileExistsError('ARCHIVE_SCENE_ALREADY_EXISTS')
    stage.rename(target)
    return {'scene':sid,'status':'ARCHIVED_AND_SHA_VERIFIED','manifest_sha256':expected['manifest.json']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--nas',type=Path,required=True)
    p.add_argument('--max-workers',type=int,default=4);a=p.parse_args()
    if not 1<=a.max_workers<=4:raise ValueError('AT_MOST_FOUR_INFERENCE_GPUS')
    a.runtime.mkdir(exist_ok=True);lock=(a.runtime/'controller.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    scope,approval=require_scope(a.root/'scope_v4_c1_TEST.json',a.root/'authorization_v4_c1_TEST.json','replay')
    fp=source_fingerprint();pre=json.loads((a.root/'v4_c1_test_preflight.json').read_text())
    if pre['status']!='PASSED' or pre['source_sha256']!=fp:raise ValueError('CURRENT_FULL_SCENE1_PREFLIGHT_REQUIRED')
    repo=Path(__file__).resolve().parents[2];script=repo/'worldmot/scripts/run_worldobject_v4_test.py'
    logs=a.root/'logs';logs.mkdir(exist_ok=True)
    env={**os.environ,'PYTHONPATH':str(repo/'worldmot'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    state={'status':'STARTING','pid':os.getpid(),'source_sha256':fp,'started_UTC':now(),
        'new_training_started':False,'TEST_parameter_selection':False,'variant':'C1',
        'hardware_reason':'Original R2 CUDA FP32 restored for exact Scene1 archived-state reproduction; CPU failed exact numeric compatibility.',
        'approval':approval,'root':str(a.root),'NAS_root':str(a.nas),'scenes':{},'archives':{}}
    def save(status=None):
        if status:state['status']=status
        state['updated_UTC']=now();write_json(a.runtime/'controller_state.json',state)
    a.nas.mkdir(parents=True,exist_ok=True)
    for name in ['scope_v4_c1_TEST.json','authorization_v4_c1_TEST.json','v4_c1_test_preflight.json','preflight_environment_attempts.json']:
        source=a.root/name;target=a.nas/name
        if target.exists() and sha256(target)!=sha256(source):raise ValueError('ARCHIVE_METADATA_CONFLICT '+name)
        if not target.exists():shutil.copyfile(source,target)
        if sha256(source)!=sha256(target):raise ValueError('ARCHIVE_METADATA_SHA '+name)
    archives=ThreadPoolExecutor(max_workers=1);copy_jobs={};workers={};failures={}
    pending=[]
    for sid in range(1,23):
        destination=a.root/f'predictions/V4_C1/scene_{sid:02d}'
        if destination.exists():
            m=completed_scene(destination,scope,sid)
            state['scenes'][str(sid)]={'status':'COMPLETE_REUSED','timestamps':m['timestamps'],'tracks_sha256':m['tracks_sha256']}
            copy_jobs[sid]=archives.submit(archive_scene,a.root,a.nas,sid)
        else:pending.append(sid)
    pending.sort(key=lambda sid:scope['scenes'][str(sid)]['candidates']**2/max(scope['scenes'][str(sid)]['timestamps'],1),reverse=True)
    available=idle_gpus()[:a.max_workers]
    state['initial_idle_GPUs']=[{'physical_index':i,'uuid':u} for i,u in available]
    save('FULL22_C1_CAUSAL_INFERENCE_RUNNING')
    try:
        while pending or workers:
            if source_fingerprint()!=fp:
                state['source_changed_hold_new_jobs']=True
            for sid,(process,gpu,handle) in list(workers.items()):
                code=process.poll()
                if code is None:continue
                handle.close();workers.pop(sid)
                if code==0:
                    m=completed_scene(a.root/f'predictions/V4_C1/scene_{sid:02d}',scope,sid)
                    state['scenes'][str(sid)].update(status='COMPLETE',timestamps=m['timestamps'],tracks_sha256=m['tracks_sha256'])
                    copy_jobs[sid]=archives.submit(archive_scene,a.root,a.nas,sid)
                else:
                    failures[sid]=code;state['scenes'][str(sid)].update(status='FAILED',exit_code=code)
                # Recheck availability; do not use a GPU acquired by another task.
                if gpu in idle_gpus():available.append(gpu)
            for sid,future in list(copy_jobs.items()):
                if future.done() and str(sid) not in state['archives']:
                    try:state['archives'][str(sid)]=future.result()
                    except Exception as e:state['archives'][str(sid)]={'status':'ARCHIVE_FAILED_LOCAL_IMMUTABLE_OUTPUT_RETAINED','error':repr(e)}
            busy={x[1][1] for x in workers.values()}
            fresh=[x for x in idle_gpus() if x[1] not in busy]
            for gpu in fresh:
                if not pending or len(workers)>=a.max_workers or state.get('source_changed_hold_new_jobs'):break
                sid=pending.pop(0);handle=(logs/f'scene_{sid:02d}.log').open('x')
                command=[sys.executable,'-u',str(script),'--stage','scene','--output',str(a.root),'--scene',str(sid),'--device','cuda:0']
                process=subprocess.Popen(command,cwd=repo,env={**env,'CUDA_VISIBLE_DEVICES':gpu[1]},
                    stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                workers[sid]=(process,gpu,handle)
                state['scenes'][str(sid)]={'status':'RUNNING','pid':process.pid,'physical_GPU':gpu[0],'GPU_UUID':gpu[1],'command':command}
            save()
            if pending and not workers and state.get('source_changed_hold_new_jobs'):
                save('SOURCE_CHANGED_NO_MORE_SCENES_LAUNCHED');return
            time.sleep(5)
        if failures:save('INCOMPLETE_FAILED_SCENES_PRESERVED');return
        coverage_log=(logs/'coverage.log').open('x')
        code=subprocess.run([sys.executable,str(script),'--stage','coverage','--output',str(a.root)],cwd=repo,env=env,
            stdin=subprocess.DEVNULL,stdout=coverage_log,stderr=subprocess.STDOUT).returncode;coverage_log.close()
        if code:save('COVERAGE_CHECK_FAILED');return
        save('ALL22_FROZEN_NATIVE_EVALUATION_RUNNING')
        output=a.root/'evaluation/V4_C1'
        evaluate=['/home/liuyeqiang/anaconda3/bin/python',str(repo/'worldmot/scripts/evaluate_fullvideo.py'),
            '--authorization',str(a.root/'authorization_v4_c1_TEST.json'),'--dataset-root','/data/DATASETS/TRACKING/JDE/VisionTrack',
            '--split','test','--audit',str(Path(scope['r2_root']).parent/'fulltrain_20261022/audit/data_audit.json'),
            '--predictions',str(a.root/'predictions/V4_C1'),'--output',str(output),'--method-id','FULL_WORLD_OBJECT_V3']
        with (logs/'official_evaluation.log').open('x') as log:
            evaluation=subprocess.Popen(evaluate,cwd=repo,env={**env,'CUDA_VISIBLE_DEVICES':''},stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        state['evaluation']={'pid':evaluation.pid,'command':evaluate}
        analysis=None
        while evaluation.poll() is None or analysis is None or analysis.poll() is None:
            compare=repo/'worldmot/analysis/compare_worldobject_v4_c1_fulltest.py'
            ready=a.runtime/'posthoc_analysis_ready.json'
            if analysis is None and ready.exists() and compare.exists():
                receipt=json.loads(ready.read_text())
                if receipt['script_sha256']!=sha256(compare):raise ValueError('POSTHOC_ANALYSIS_SOURCE_CHANGED')
                with (logs/'identity_attribution.log').open('x') as log:
                    command=[sys.executable,'-u',str(compare),'--root',str(a.root),'--r2-root',scope['r2_root']]
                    analysis=subprocess.Popen(command,cwd=repo,env={**env,'CUDA_VISIBLE_DEVICES':''},stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                state['posthoc']={'pid':analysis.pid,'command':command,'script_sha256':receipt['script_sha256']}
            state['evaluation_exit_code']=evaluation.poll()
            state['posthoc_exit_code']=None if analysis is None else analysis.poll()
            save();time.sleep(10)
        for sid,future in copy_jobs.items():
            try:state['archives'][str(sid)]=future.result()
            except Exception as e:state['archives'][str(sid)]={'status':'ARCHIVE_FAILED_LOCAL_IMMUTABLE_OUTPUT_RETAINED','error':repr(e)}
        save('FULL22_INFERENCE_AND_NATIVE_EVAL_POSTHOC_FINISHED_REVIEW_REQUIRED')
    finally:
        archives.shutdown(wait=True)


if __name__=='__main__':main()
