"""Run the fixed36 first-seed native matrix after owned admission/engineering.

No model/threshold/scene search. Existing root credit and all old costs retained.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common import ROOT,save_json,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--actor-source',type=Path,required=True);p.add_argument('--ledger',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True);rt=a.runtime;state=dict(status='WAIT_EXISTING_OWNED_QUALIFICATIONS',pid=os.getpid(),scientific_success=False)
    start=time.monotonic();queue=rt/'cow_complete_v1_queue/STATE.json'
    while True:
        ledger=json.loads(a.ledger.read_text());running=[s for s in ledger['stages'] if s['status'].startswith('RUNNING')]
        for s in running:
            if not Path('/proc/'+str(s['supervisor_pid'])).exists():raise RuntimeError('Missing active supervisor, no inferred completion')
        q=json.loads(queue.read_text()) if queue.exists() else None
        queue_terminal=q and q['status'] in ('COMPLETE','FAILED_EVIDENCE_RETAINED')
        if not queue_terminal and q and not Path('/proc/'+str(q['pid'])).exists():raise RuntimeError('Missing queued engineering supervisor')
        state.update(wait_seconds=time.monotonic()-start,active_stages=[s['stage'] for s in running]);save_json(a.out/'STATE.json',state)
        if not running and queue_terminal:break
        if time.monotonic()-start>21660:raise TimeoutError('Bounded admission wait expired')
        time.sleep(15)
    preflight=rt/'p4_native_preflight_v1';proof=json.loads((preflight/'RESULT.json').read_text())
    if proof['status']!='PASS_P4_NATIVE_PREFIXES_FROZEN_HEAD_FAULTS_AND_COMPLETE_OFF':raise ValueError('P4 admission failed; no automatic retry')
    destination=rt/'p4_development_v1'
    subprocess.run([sys.executable,'geo_tools/prepare_cir_p4_development.py','--gate',str(preflight/'full_online_gate.json'),
        '--source-root',str(a.actor_source),'--out',str(destination),'--gpus','6','7','8','9'],cwd=ROOT,check=True)
    allocation=destination/'ALLOCATION.json';state.update(status='RUNNING_REGISTERED_DEV36',allocation_sha256=sha256(allocation));save_json(a.out/'STATE.json',state)
    with (a.out/'batch.log').open('x') as log:
        process=subprocess.Popen([sys.executable,'-u','geo_tools/run_cir_batch.py','--allocation',str(allocation),'--sha256',sha256(allocation),'--ledger',str(a.ledger)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        state['batch_pid']=process.pid;save_json(a.out/'STATE.json',state);code=process.wait()
    state.update(batch_exit_code=code,status='BATCH_COMPLETE' if code==0 else 'BATCH_FAILED_EVIDENCE_RETAINED',evaluations={});save_json(a.out/'STATE.json',state)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    for mode in ('guard_reference','risk_mlp','risk_grouped'):
        runs=destination/(mode+'_runs.json');spec=json.loads(runs.read_text())
        complete=all(Path(p).exists() and json.loads(Path(p).read_text()).get('status')=='COMPLETE' for p in spec['runs'].values())
        if not complete:
            state['evaluations'][mode]={'status':'NOT_COMPLETE_ALL12_NO_POOLED_SUBSET_SCORE'};continue
        with (a.out/(mode+'_evaluation.log')).open('x') as log:
            result=subprocess.run([sys.executable,'-u','geo_tools/evaluate_cir_fulltrain.py','--runs',str(runs),'--out',str(rt/f'p4_dev_{mode}_eval_v1')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        state['evaluations'][mode]={'exit_code':result.returncode};save_json(a.out/'STATE.json',state)
    state['status']='COMPLETE_PENDING_RESEARCH_REVIEW' if code==0 and all(x.get('exit_code')==0 for x in state['evaluations'].values()) else 'INCOMPLETE_EVIDENCE_RETAINED'
    save_json(a.out/'STATE.json',state)


if __name__=='__main__':main()
