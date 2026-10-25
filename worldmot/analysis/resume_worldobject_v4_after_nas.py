#!/usr/bin/env python3
"""User-authorized deferred finite experiments after NAS power restoration.

No training or V4 online TEST entry point exists here. Healthy workers are never
terminated/restarted. Inputs and R2/R3 outputs are read-only. Changes of source,
failed preflight or failed workers stop dependent work and retain their logs.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from worldobject_v4_audit_common import VAL_SCENES,code_fingerprint,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--r2-root',type=Path,required=True)
    p.add_argument('--not-before',default='2026-10-25T20:00:00+00:00');a=p.parse_args()
    repo=Path(__file__).resolve().parents[2];a.runtime.mkdir(parents=True,exist_ok=True)
    lock=(a.runtime/'controller.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('V4 controller already running; no duplicate')
    receipt=json.loads((a.runtime/'cpu_receipt.json').read_text());fp=code_fingerprint(repo)
    if receipt['status']!='PASSED' or receipt['source_sha256']!=fp:raise ValueError('CURRENT_CPU_PREFLIGHT_REQUIRED')
    not_before=datetime.fromisoformat(a.not_before).timestamp()
    state={'source_sha256':fp,'pid':os.getpid(),'not_before_UTC':a.not_before,'user_reference':
        'NAS power outage; user explicitly said continue independent work now and resume after 04:00 Asia/Shanghai',
        'new_training_authorized':False,'new_V4_formal_TEST_authorized':False,'started_UTC':datetime.now(timezone.utc).isoformat()}
    def status(value,**extra):
        state.update(status=value,updated_UTC=datetime.now(timezone.utc).isoformat(),**extra)
        write_json(a.runtime/'controller_state.json',state);print(json.dumps(state),flush=True)
    status('WAITING_UNTIL_USER_NAS_RECOVERY_WINDOW')
    while time.time()<not_before:
        if (a.runtime/'STOP_BEFORE_NEW_WORK').exists():status('STOPPED_BEFORE_LAUNCH');return
        if code_fingerprint(repo)!=fp:status('HELD_SOURCE_CHANGED_NO_LAUNCH');return
        time.sleep(min(45,max(1,not_before-time.time())))
    status('WAITING_FOR_NAS_IMMUTABLE_INPUTS')
    while True:
        if (a.runtime/'STOP_BEFORE_NEW_WORK').exists():status('STOPPED_BEFORE_LAUNCH');return
        if code_fingerprint(repo)!=fp:status('HELD_SOURCE_CHANGED_NO_LAUNCH');return
        # A separate bounded probe prevents a stalled FUSE stat from blocking
        # the controller forever. It reads an existing signed manifest first.
        command=[sys.executable,'-c',
            'from pathlib import Path; import sys,json,shutil; p=Path(sys.argv[1]); '
            'g=json.loads((p/"train/FULL_WORLD_OBJECT_V3/train_gate.json").read_text()); '
            'assert g["status"]=="FULL_TRAIN_GATE_PASSED"; '
            'print(json.dumps({"free_bytes":shutil.disk_usage(p).free}))',str(a.r2_root)]
        try:
            result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=12)
            if result.returncode==0:
                free=json.loads(result.stdout)['free_bytes']
                if free<25*(1<<30):status('HELD_NAS_FREE_SPACE_BELOW_25_GIB',NAS_free_bytes=free);return
                break
        except (subprocess.TimeoutExpired,OSError,json.JSONDecodeError):pass
        time.sleep(45)
    if code_fingerprint(repo)!=fp:status('HELD_SOURCE_CHANGED_NO_LAUNCH');return
    root=a.r2_root.parent/'worldobject_v4_persistent_state'
    root.mkdir(parents=True,exist_ok=False)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
         'PYTHONPATH':str(repo/'worldmot')}
    def run(name,arguments,environment=env):
        log=(root/(name+'.log')).open('x')
        process=subprocess.Popen([sys.executable,'-u',*arguments],cwd=repo,env=environment,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        log.close();return process
    replay=str(Path(__file__).with_name('run_worldobject_v4_train_replay.py'))
    status('REAL_CPU_PREFLIGHT_RUNNING',run_root=str(root),NAS_free_bytes=free)
    pre=run('CPU_real_preflight',[replay,'--r2-root',str(a.r2_root),'--output',str(root/'CPU_preflight'),
        '--scene','4','--limit','16','--with-single-step'])
    code=pre.wait()
    if code:status('REAL_PREFLIGHT_FAILED_LOGS_PRESERVED',exit_code=code);return
    preflight=json.loads((root/'CPU_preflight/replay_report.json').read_text())
    if preflight['C0_R2_exact_frames']!=16:status('REAL_PREFLIGHT_C0_EXACTNESS_FAILED');return
    if code_fingerprint(repo)!=fp:status('HELD_SOURCE_CHANGED_AFTER_PREFLIGHT');return
    status('TRAIN_SIX_SCENES_AND_EXISTING_TEST_SYNC_AUDIT_RUNNING',replay_device='cpu',healthy_workers_will_not_be_restarted=True)
    workers={sid:run(f'TRAIN_scene_{sid:02d}',[replay,'--r2-root',str(a.r2_root),'--output',str(root/f'scene_{sid:02d}'),
        '--scene',str(sid),'--with-single-step']) for sid in VAL_SCENES}
    stage_a=run('existing_TEST_sync_audit',[str(Path(__file__).with_name('compare_gmt_v4_synchronous_identity.py')),
        '--r2-root',str(a.r2_root),'--output',str(root/'v4_gmt_synchronous_identity_comparison.json'),'--with-current-only'])
    mechanism=run('retained_original_case',[replay,'--r2-root',str(a.r2_root),'--output',str(root/'retained_case'),
        '--scene','24','--limit','705','--scope','state','--retained-mechanism'])
    failures={}
    while any(worker.poll() is None for worker in workers.values()):
        if code_fingerprint(repo)!=fp:
            # Existing healthy jobs continue; only dependent launches are held.
            state['source_changed_while_jobs_running']=True
        state['TRAIN_workers']={str(sid):{'pid':w.pid,'exit_code':w.poll()} for sid,w in workers.items()}
        write_json(a.runtime/'controller_state.json',state);time.sleep(20)
    failures={str(sid):w.returncode for sid,w in workers.items() if w.returncode}
    # A failure cannot cancel independent authorized audits or healthy workers.
    a_code=stage_a.wait();mechanism_code=mechanism.wait()
    if a_code:
        write_json(root/'v4_gmt_synchronous_identity_comparison.json',{'status':'AUDIT_UNAVAILABLE_LOGS_PRESERVED',
            'exit_code':a_code,'GMT_sync_metrics':None,'inference_or_thresholds_not_modified':True})
    if failures:status('TRAIN_REPLAY_FAILURE_NO_AUTOMATIC_RESTART',failed_scenes=failures,
        stage_A_exit=a_code,retained_case_exit=mechanism_code);return
    if code_fingerprint(repo)!=fp:status('HELD_SOURCE_CHANGED_NO_DEPENDENT_SCORING');return
    status('FULL_TRAIN_VAL_NATIVE_SCORING_RUNNING',stage_A_exit=a_code,retained_case_exit=mechanism_code)
    scorer=run('native_TRAIN_VAL',[str(Path(__file__).with_name('score_worldobject_v4_train_val.py')),
        '--root',str(root),'--r2-root',str(a.r2_root)])
    if scorer.wait():status('NATIVE_TRAIN_VAL_SCORING_FAILED_LOGS_PRESERVED',exit_code=scorer.returncode);return
    writer=str(Path(__file__).with_name('write_worldobject_v4_report.py'))
    report=subprocess.run([sys.executable,writer,'--runtime',str(a.runtime),'--cpu-junit',receipt['junit'],
        '--measurements',str(root)],cwd=repo,env=env)
    if report.returncode:status('REPORT_FAILED_LOGS_PRESERVED');return
    gate=json.loads((repo/'worldmot/results/v4_stage_gate.json').read_text())
    # Limited LIVE engineering only, on a genuinely idle authorized V100.
    query=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],
        text=True,stdout=subprocess.PIPE,check=True)
    idle=[]
    for line in query.stdout.splitlines():
        index,uuid,memory=[x.strip() for x in line.split(',')]
        if int(index) in (4,5,6,7,8,9) and int(memory)<100:idle.append((index,uuid))
    if idle:
        gpu_index,gpu_uuid=idle[0];variants=['C0','C1','C2','C3','C4']+(['C5'] if gate.get('C5_live_quality_gate_passed') else [])
        live_env={**env,'CUDA_VISIBLE_DEVICES':gpu_uuid}
        arguments=[str(Path(__file__).with_name('run_worldobject_v4_live_preflight.py')),'--r2-root',str(a.r2_root),
            '--output',str(root/'LIVE_TRAIN_preflight'),'--variants',*variants,'--limit','30']
        if 'C5' in variants:arguments+=['--C5-quality-gate',str(repo/'worldmot/results/v4_stage_gate.json')]
        status('FINITE_REAL_TRAIN_LIVE_TIMING_RUNNING',physical_GPU=gpu_index,variants=variants)
        live=run('LIVE_TRAIN_preflight',arguments,live_env);live_code=live.wait()
        state['live_exit_code']=live_code
        if live_code==0:
            runtime=json.loads((root/'v4_runtime_analysis.json').read_text())
            runtime['finite_real_TRAIN_LIVE']=json.loads((root/'LIVE_TRAIN_preflight/v4_live_runtime.json').read_text())
            runtime['status']='TRAIN_CACHE_AND_FINITE_REAL_LIVE_COMPLETE'
            runtime['C5_live_status']='MEASURED' if 'C5' in variants else 'NOT_RUN_C4_QUALITY_GATE_FAILED'
            write_json(root/'v4_runtime_analysis.json',runtime)
    else:state['live_status']='NOT_RUN_NO_IDLE_AUTHORIZED_GPU'
    subprocess.run([sys.executable,writer,'--runtime',str(a.runtime),'--cpu-junit',receipt['junit'],
        '--measurements',str(root)],cwd=repo,env=env,check=True)
    status('FINITE_EXPERIMENTS_FINISHED_READY_FOR_REVIEW_AND_GITHUB',new_training_started=False,
        new_V4_formal_TEST_started=False,run_root=str(root))


if __name__=='__main__':main()
