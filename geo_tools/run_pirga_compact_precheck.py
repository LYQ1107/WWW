"""Actual40-frame compact collector parity; one assigned-GPU process, bounded."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from pirga_indexed_store import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--permit',type=Path,required=True);p.add_argument('--permit-sha',required=True);p.add_argument('--off',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--python',type=Path,required=True)
    a=p.parse_args()
    if sha(a.permit)!=a.permit_sha:raise ValueError('Unpinned precheck allocation')
    permit=json.loads(a.permit.read_text());parent=permit['parent_budget']
    if (permit['status']!='ALLOCATED_WITHIN_EXISTING_CAP' or permit['maximum_total_GPU_wall_hours']>.1 or a.gpu in (0,1,4,5,6)
        or sha(parent['path'])!=parent['sha256'] or parent['recorded_usage']+permit['previous_PIRGA_actual_GPU_hours']+permit['maximum_total_GPU_wall_hours']>parent['cap']):raise ValueError('Scope/quota mismatch')
    used=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    if used:raise ValueError('Device is occupied')
    a.out.mkdir(parents=True,exist_ok=False)
    cmd=[str(a.python),'-u',str(Path(__file__).with_name('export_pirga_full_scene.py')),'--permit',str(a.permit),'--permit-sha',a.permit_sha,'--scene','4','--prefix40','--out',str(a.out/'on')]
    started=time.monotonic();child=None;ledger=dict(status='RUNNING',command=cmd,gpu=a.gpu,permit_sha256=a.permit_sha)
    try:
        with open(a.out/'on.log','x') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),start_new_session=True)
            ledger['pid']=child.pid
            (a.out/'RESOURCE_LEDGER.json').write_text(json.dumps(ledger,indent=2)+'\n')
            code=child.wait(timeout=min(permit['maximum_single_process_seconds'],permit['maximum_total_GPU_wall_hours']*3600-5))
            if code:raise RuntimeError('Compact collector failed '+str(code))
        # Original frozen OFF stream has no new score controller.
        for on_name,off_name in [('predictions.json','predictions.json'),('first40_group_state_fingerprints.json','group_state_fingerprints.json')]:
            if (a.out/'on'/on_name).read_bytes()!=(a.off/off_name).read_bytes():raise ValueError('Actual OFF/compactON parity failed '+on_name)
        gate=dict(status='PASS_REAL40_COMPACT_COLLECTOR_OFF_ON',scene=4,groups=40,original_prediction_and_Gallery_Bank_state_bytes_exact=True,
                  on_manifest_sha256=sha(a.out/'on/manifest.json'),off_manifest_sha256=sha(a.off/'manifest.json'),scope_sha256=permit['scope_sha256'],
                  full11_data_admitted=False,GT_semantics_loaded=False)
        (a.out/'GATE.json').write_text(json.dumps(gate,indent=2)+'\n');ledger['status']='PASS'
    except BaseException as e:
        ledger.update(status='FAILED_RETAIN_EVIDENCE',reason=repr(e));raise
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        ledger['used_GPU_wall_hours']=(time.monotonic()-started)/3600
        (a.out/'RESOURCE_LEDGER.json').write_text(json.dumps(ledger,indent=2)+'\n')

if __name__=='__main__':main()
