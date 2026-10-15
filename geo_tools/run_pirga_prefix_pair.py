"""One-shot authorized OFF/ON native probe with aggregate assigned-GPU wall cap."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from export_pirga_native_prefix import validate_permit
from pirga_indexed_store import sha


def save(path,body):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(body,indent=2)+'\n');tmp.replace(path)


def run(permit_path,permit_sha,out,gpu,python):
    permit=validate_permit(permit_path,permit_sha,4,40)
    if gpu in (0,4,5,6):raise ValueError('Other research occupied/reserved GPUs excluded')
    # The permit is consumed once, not reusable to bypass the aggregate budget.
    out.mkdir(parents=True,exist_ok=False)
    lock=Path(permit['one_shot_lock_path'])
    with lock.open('x') as f:f.write(str(out.resolve()))
    ledger=dict(status='PREFLIGHT',permit_sha256=permit_sha,maximum_gpu_wall_hours=permit['maximum_total_gpu_wall_hours'],used_gpu_wall_hours=0.,gpu=gpu,processes=[],old_ledgers_reset=False,
                started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    lp=out/'RESOURCE_LEDGER.json';save(lp,ledger)
    try:
        if permit.get('authorized_gpu_indices') is not None and gpu not in permit['authorized_gpu_indices']:raise ValueError('GPU outside permit')
        for mode in ('off','on'):
            used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
            if used!=0:raise RuntimeError('GPU acquired by another job; no launch')
            remaining=permit['maximum_total_gpu_wall_hours']*3600-ledger['used_gpu_wall_hours']*3600
            limit=min(permit['maximum_process_seconds'],remaining-5)
            if limit<=30:raise RuntimeError('Insufficient remaining authorized time')
            cmd=[str(python),'-u',str(Path(__file__).with_name('export_pirga_native_prefix.py')),'--permit',str(permit_path),'--permit-sha',permit_sha,'--export',mode,'--out',str(out/mode)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            started=time.monotonic();child=None
            row=dict(mode=mode,command=cmd,status='STARTING',assigned_gpu=gpu);ledger['processes'].append(row)
            with open(out/(mode+'.log'),'x') as log:
                try:
                    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                    row.update(pid=child.pid,status='RUNNING');ledger['status']='RUNNING';save(lp,ledger)
                    code=child.wait(timeout=limit)
                    if code:raise RuntimeError('Export process failed: '+str(code))
                    row['status']='PASS'
                except BaseException:
                    row['status']='FAILED_OR_BUDGET_TERMINATED'
                    if child and child.poll() is None:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=3)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    raise
                finally:
                    row['wall_seconds']=time.monotonic()-started;ledger['used_gpu_wall_hours']+=row['wall_seconds']/3600;save(lp,ledger)
        for name in ('predictions.json','group_state_fingerprints.json'):
            if (out/'off'/name).read_bytes()!=(out/'on'/name).read_bytes():raise RuntimeError('Real native OFF/ON parity failed: '+name)
        on=json.loads((out/'on/manifest.json').read_text())
        save(out/'GATE.json',dict(status='PASS_REAL_FIT40_OFF_ON_NATIVE_EXPORT',scene=4,groups=40,original_outputs_and_state_fingerprints_byte_exact=True,
             on_manifest_sha256=sha(out/'on/manifest.json'),off_manifest_sha256=sha(out/'off/manifest.json'),queries=on['queries'],calls=on['calls'],
             gpu_wall_hours=ledger['used_gpu_wall_hours'],full11_admitted=False))
        ledger['status']='COMPLETE_PREFIX_ONLY'
    except BaseException as e:
        ledger.update(status='FAILED_RETAIN_EVIDENCE',reason=repr(e));raise
    finally:save(lp,ledger)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--permit',type=Path,required=True);p.add_argument('--permit-sha',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--python',type=Path,required=True)
    a=p.parse_args();run(a.permit,a.permit_sha,a.out,a.gpu,a.python)
