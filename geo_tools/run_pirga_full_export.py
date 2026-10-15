"""Finite idle-GPU queue with aggregate actual process wall cap and compact ledger."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from pirga_indexed_store import sha


def save(path,body):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(body,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--permit',type=Path,required=True);p.add_argument('--permit-sha',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpus',type=int,nargs='+',required=True);p.add_argument('--python',type=Path,required=True);a=p.parse_args()
    if sha(a.permit)!=a.permit_sha:raise ValueError('Unpinned stage allocation')
    permit=json.loads(a.permit.read_text());scope=json.loads(Path(permit['scope_path']).read_text())
    if permit['status']!='ALLOCATED_WITHIN_EXISTING_CAP' or sha(permit['scope_path'])!=permit['scope_sha256']:raise ValueError('Invalid scope/allocation')
    if set(a.gpus)&{0,1,4,5,6} or len(set(a.gpus))!=len(a.gpus):raise ValueError('Other project/occupied GPUs excluded')
    parent=permit['parent_budget']
    if sha(parent['path'])!=parent['sha256'] or parent['recorded_usage']+permit['previous_PIRGA_actual_GPU_hours']+permit['maximum_total_GPU_wall_hours']>parent['cap']:raise ValueError('Parent cap changed or exceeded')
    gate=Path(permit['compact_collector_gate_path'])
    if sha(gate)!=permit['compact_collector_gate_sha256'] or json.loads(gate.read_text())['status']!='PASS_REAL40_COMPACT_COLLECTOR_OFF_ON':raise ValueError('New collector actual gate required')
    a.out.mkdir(parents=True,exist_ok=False);rows=[];live={};completed_seconds=0.;budget=permit['maximum_total_GPU_wall_hours']*3600
    scenes=sorted(scope['scenes'],key=lambda s:-s['historical_native_seconds']);pending=[s['scene'] for s in scenes]
    ledger=dict(status='RUNNING',permit_sha256=a.permit_sha,maximum_GPU_wall_hours=permit['maximum_total_GPU_wall_hours'],used_GPU_wall_hours=0.,jobs=rows,
                pending_scenes=pending,old_ledgers_reset=False,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());lp=a.out/'RESOURCE_LEDGER.json'
    def spent():return completed_seconds+sum(time.monotonic()-j['start'] for j in live.values())
    def stop(j):
        child=j['process']
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    try:
        while pending or live:
            if sha(parent['path'])!=parent['sha256']:raise RuntimeError('Parent quota source changed; retain all evidence')
            if spent()>budget-20*max(1,len(live)):raise RuntimeError('Aggregate assigned-GPU wall cap reached')
            if sum(f.stat().st_size for f in a.out.rglob('*') if f.is_file())>permit['maximum_total_output_bytes']:raise RuntimeError('Aggregate output cap reached')
            for gpu in a.gpus:
                if gpu in live or not pending:continue
                memory=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
                if memory:continue
                scene=pending.pop(0);out=a.out/f'scene{scene}';cmd=[str(a.python),'-u',str(Path(__file__).with_name('export_pirga_full_scene.py')),'--permit',str(a.permit),'--permit-sha',a.permit_sha,'--scene',str(scene),'--out',str(out)]
                log=open(a.out/f'scene{scene}.log','x');start=time.monotonic();env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                row=dict(scene=scene,gpu=gpu,pid=child.pid,status='RUNNING',command=cmd);rows.append(row);live[gpu]=dict(process=child,log=log,start=start,row=row)
                print(json.dumps(dict(start_scene=scene,gpu=gpu,pid=child.pid)),flush=True)
            for gpu,j in list(live.items()):
                elapsed=time.monotonic()-j['start'];code=j['process'].poll()
                if code is None:
                    if elapsed>permit['maximum_single_process_seconds']:raise RuntimeError('Scene process cap reached')
                    continue
                j['log'].close();completed_seconds+=elapsed;j['row'].update(status='PASS' if code==0 else 'FAILED',exit_code=code,wall_seconds=elapsed);del live[gpu]
                if code:raise RuntimeError('Scene failed '+str(j['row']['scene']))
                m=json.loads((a.out/f"scene{j['row']['scene']}"/'manifest.json').read_text())
                if m['status']!='COMPLETE_FULL_NATIVE_INPUTS_PARITY_PASS_PENDING_SUPERVISION':raise RuntimeError('Incomplete scene cannot be accepted')
                print(json.dumps(dict(done_scene=j['row']['scene'],wall_seconds=elapsed)),flush=True)
            ledger.update(used_GPU_wall_hours=spent()/3600,pending_scenes=list(pending),live_pids=[j['process'].pid for j in live.values()]);save(lp,ledger)
            if pending or live:time.sleep(2)
        ledger.update(status='ALL11_CAPTURED_PENDING_CONSUMER_ADMISSION',live_pids=[])
    except BaseException as e:
        ledger.update(status='FAILED_RETAIN_PARTIAL_NO_ADMISSION',reason=repr(e));raise
    finally:
        for j in live.values():
            stop(j);j['log'].close();j['row'].update(status='TERMINATED_AFTER_STAGE_FAILURE',wall_seconds=time.monotonic()-j['start']);completed_seconds+=j['row']['wall_seconds']
        ledger.update(used_GPU_wall_hours=completed_seconds/3600,live_pids=[]);save(lp,ledger)

if __name__=='__main__':main()
