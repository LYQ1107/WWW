"""Resume the unchanged finite matrix only after the ledger fix qualifies.

Old failed artifacts/costs stay intact. This attempt reuses existing CIR credit,
and runs unmodified official metrics only after all48 complete online runs.
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
    a.out.mkdir(parents=True);r=a.runtime;ledger_path=ROOT/'results/CIR_GMT/RESOURCE_LEDGER.json'
    state=dict(status='WAITING_VERIFIED_LEDGER_FIX_QUALIFICATION',pid=os.getpid(),scientific_success=False)
    save_json(a.out/'STATE.json',state);start=time.monotonic()
    while True:
        ledger=json.loads(ledger_path.read_text());qualification=next(s for s in ledger['stages'] if s['stage']=='P2_LEDGER_REPAIR_FULL_NATIVE_QUALIFICATION_V2')
        if not qualification['status'].startswith('RUNNING'):break
        if not Path('/proc/'+str(qualification['supervisor_pid'])).exists():raise RuntimeError('Qualification supervisor missing; no inferred completion or restart')
        if time.monotonic()-start>3600:raise TimeoutError('Bounded qualification wait expired')
        state['wait_seconds']=time.monotonic()-start;save_json(a.out/'STATE.json',state);time.sleep(15)
    if qualification['status']!='COMPLETE':raise RuntimeError('Ledger fix failed qualification; no full12 replay')
    gate=r/'p2_finite_full_online_preflight_v2/full_online_gate.json';data=json.loads(gate.read_text())
    if data['allowed_methods']!=['off','guard','margin','threshold','bias2']:raise ValueError('All fixed controls must qualify')
    source=Path(json.loads((r/'p2_qualification_v2/ALLOCATION.json').read_text())['source_root'])
    destination=r/'p2_development_v2'
    subprocess.run([sys.executable,'geo_tools/prepare_cir_p2_development.py','--gate',str(gate),
        '--source-root',str(source),'--out',str(destination),'--gpus','6','7','8','9','--attempt','v2'],cwd=ROOT,check=True)
    allocation=destination/'ALLOCATION.json';alloc=json.loads(allocation.read_text());ledger=json.loads(ledger_path.read_text())
    if any(s['status'].startswith('RUNNING') or s['stage']==alloc['stage'] for s in ledger['stages']):raise ValueError('Concurrent writer or repeated attempt')
    if ledger['measured_GPU_wall_hours']+12>ledger['allocated_total_GPU_wall_hours']:raise ValueError('Existing CIR credit insufficient; no silent credit reset')
    ledger.setdefault('scope_allocations',[]).append(dict(stage=alloc['stage'],maximum_GPU_wall_hours=12.,
        source_of_budget='existing registered CIR credit; retain all v1 and qualification actual costs; new credit0',
        allocation_sha256=sha256(allocation)))
    save(ledger_path,ledger)
    state.update(status='FULL12_V2_RUNNING',source_commit=alloc['source_commit'],allocation_sha256=sha256(allocation))
    save_json(a.out/'STATE.json',state)
    with (a.out/'full12_batch.log').open('x') as log:
        process=subprocess.Popen([sys.executable,'-u','geo_tools/run_cir_batch.py','--allocation',str(allocation),
            '--sha256',sha256(allocation),'--ledger',str(ledger_path)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        state['batch_pid']=process.pid
        while process.poll() is None:save_json(a.out/'STATE.json',state);time.sleep(15)
    if process.returncode:
        state.update(status='FAILED_FULL12_V2_EVIDENCE_RETAINED',batch_exit_code=process.returncode);save_json(a.out/'STATE.json',state)
        raise RuntimeError('Full12 v2 failed; do not hide or automatically retry')
    state.update(status='FULL12_V2_COMPLETE_EVALUATING',evaluations={});save_json(a.out/'STATE.json',state)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    for method in ('guard','margin','threshold','bias2'):
        output=r/f'p2_dev_{method}_eval_v2'
        with (a.out/(method+'_evaluation.log')).open('x') as log:
            evaluated=subprocess.run([sys.executable,'-u','geo_tools/evaluate_cir_fulltrain.py','--runs',
                str(destination/(method+'_runs.json')),'--out',str(output)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        state['evaluations'][method]=dict(exit_code=evaluated.returncode,result=str(output/'RESULT.json'))
        save_json(a.out/'STATE.json',state)
    state['status']='COMPLETE_PENDING_RESEARCH_REVIEW' if all(v['exit_code']==0 for v in state['evaluations'].values()) else 'OFFLINE_EVALUATION_FAILED_ONLINE_OUTPUTS_RETAINED'
    save_json(a.out/'STATE.json',state)


if __name__=='__main__':main()
