"""Continue the already frozen P2 stages when current native pairs finish.

No policy/threshold selection: terminal cost import, bounded full12 launch,
then a separate all26 offline GT evaluation while full videos are running.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common import ROOT,save_json,sha256
from run_cir_guarded import save


def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True);r=a.runtime;parent=ROOT/'results/CIR_GMT/RESOURCE_LEDGER.json'
    child=ROOT/'results/CIR_GMT/P2_PREFLIGHT_RESOURCE_LEDGER.json'
    allocation=r/'p2_development_v1/ALLOCATION.json';prepared=ROOT/'results/CIR_GMT/P2_FULL_ONLINE_PREPARATION.json'
    admission=json.loads(prepared.read_text())
    if sha256(allocation)!=admission['allocation']['sha256']:raise ValueError('Frozen full12 allocation drift')
    alloc=json.loads(allocation.read_text())
    if alloc['maximum_GPU_wall_hours']!=12 or len(alloc['jobs'])!=48 or len(alloc['physical_GPUs'])>4:
        raise ValueError('Only the registered bounded matrix is allowed')
    state=dict(status='WAITING_FOR_VERIFIED_PAIRED_PROCESS',pipeline_pid=os.getpid(),GT_in_inference=False,
        policy_selection=False,allocation_sha256=sha256(allocation));save_json(a.out/'STATE.json',state)
    begin=time.monotonic()
    while True:
        ledger=json.loads(parent.read_text())
        stage=next(s for s in ledger['stages'] if s['stage']=='P2_ALL26_PAIRED_NATIVE_FUTURES_V1')
        if not stage['status'].startswith('RUNNING'):break
        if not Path('/proc/'+str(stage['supervisor_pid'])).exists():
            raise RuntimeError('Paired supervisor handle missing; do not infer completion or restart')
        if time.monotonic()-begin>3600:raise TimeoutError('Bounded continuation wait expired; existing work untouched')
        state['wait_seconds']=time.monotonic()-begin;save_json(a.out/'STATE.json',state)
        print(json.dumps(dict(status=state['status'],seconds=state['wait_seconds'])),flush=True)
        time.sleep(15)
    if stage['status']!='COMPLETE':raise RuntimeError('Paired batch did not complete; no full12 admission')
    for scene in (12,20,21,24):
        manifest=json.loads((r/f'p2_paired_native_v1/scene{scene}/manifest.json').read_text())
        if manifest['status']!='COMPLETE_ALL_PREDECLARED_SCENE_TARGETS':raise RuntimeError('Incomplete paired scene')
    subprocess.run([sys.executable,'geo_tools/merge_cir_subledger.py','--parent',str(parent),'--child',str(child)],cwd=ROOT,check=True)
    ledger=json.loads(parent.read_text());stage_name=alloc['stage']
    if any(s['status'].startswith('RUNNING') or s['stage']==stage_name for s in ledger['stages']):raise ValueError('Resource writer conflict or repeated matrix')
    if any(s['stage']==stage_name for s in ledger.get('allocations',[])):raise ValueError('Full12 credit already registered')
    ledger['allocated_total_GPU_wall_hours']+=12.
    ledger['allocations'].append(dict(stage=stage_name,added_GPU_wall_hours=12.,path=str(allocation),
        sha256=sha256(allocation),reason=alloc['estimate'],user_request_sha256=json.loads((ROOT/'results/CIR_GMT/RESEARCH_GOAL.json').read_text())['request_sha256']))
    save(parent,ledger)
    command=[sys.executable,'-u','geo_tools/run_cir_batch.py','--allocation',str(allocation),
        '--sha256',sha256(allocation),'--ledger',str(parent)]
    log=a.out/'full12_batch.log'
    with log.open('x') as stream:
        batch=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        state.update(status='FULL12_RUNNING_GT_EVALUATION_STARTING',batch_pid=batch.pid,batch_log=str(log))
        save_json(a.out/'STATE.json',state)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with (a.out/'paired_offline_evaluation.log').open('x') as evaluation_log:
            evaluated=subprocess.run([sys.executable,'-u','geo_tools/evaluate_cir_guard_paired.py',
                '--runs',str(r/'p2_paired_native_v1'),'--events',str(r/'p2_guard_fit12_audit_v1/FIT_POLICY_EVENTS.jsonl.gz'),
                '--out',str(r/'p2_paired_native_eval_v1')],cwd=ROOT,env=env,stdout=evaluation_log,stderr=subprocess.STDOUT)
        state.update(status='FULL12_RUNNING',paired_evaluation_exit_code=evaluated.returncode)
        save_json(a.out/'STATE.json',state)
        # A formatting/evaluation failure does not cancel valid fixed-policy
        # jobs or trigger any policy selection. Keep its separate failure log.
        while batch.poll() is None:
            save_json(a.out/'STATE.json',state);time.sleep(15)
        state.update(status='FULL12_COMPLETE_PENDING_RESEARCH_EVALUATION' if batch.returncode==0 else 'FULL12_FAILED_EVIDENCE_RETAINED',
            batch_exit_code=batch.returncode)
        save_json(a.out/'STATE.json',state)
        if batch.returncode:raise RuntimeError('Owned full12 batch failed; do not restart or conceal partial runs')


if __name__=='__main__':main()
