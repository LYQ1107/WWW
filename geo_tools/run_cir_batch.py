"""Bounded registered jobs, up to four idle GPUs; one durable resource writer.

Only child process groups launched here may be terminated. Other users' jobs
are never touched. Allocation and source HEAD are pinned before any launch.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from run_cir_guarded import save


def main():
    p=argparse.ArgumentParser();p.add_argument('--allocation',type=Path,required=True)
    p.add_argument('--sha256',required=True);p.add_argument('--ledger',type=Path,required=True)
    a=p.parse_args()
    if hashlib.sha256(a.allocation.read_bytes()).hexdigest()!=a.sha256:raise ValueError('Unpinned CIR allocation')
    allocation=json.loads(a.allocation.read_text());ledger=json.loads(a.ledger.read_text())
    gpus=allocation['physical_GPUs'];budget=float(allocation['maximum_GPU_wall_hours'])
    if not 1<=len(set(gpus))==len(gpus)<=4 or budget<=0:raise ValueError('Invalid bounded allocation')
    if any(s['status'].startswith('RUNNING') for s in ledger['stages']):raise ValueError('Another CIR resource writer is active')
    if any(s['stage']==allocation['stage'] for s in ledger['stages']):raise ValueError('Never overwrite an existing stage')
    base=ledger['measured_GPU_wall_hours']
    if base+budget>ledger['allocated_total_GPU_wall_hours']:raise ValueError('CIR registered total exceeded')
    source=Path(allocation['source_root']);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    if head!=allocation['source_commit'] or subprocess.check_output(['git','status','--porcelain'],cwd=source):
        raise ValueError('Immutable clean source required')
    jobs=[dict(job,status='PENDING') for job in allocation['jobs']]
    if not jobs or len({j['name'] for j in jobs})!=len(jobs):raise ValueError('Finite unique jobs required')
    stage=dict(stage=allocation['stage'],status='RUNNING_BATCH',allocation_sha256=a.sha256,
        maximum_GPU_wall_hours=budget,physical_GPUs=gpus,jobs=jobs,supervisor_pid=os.getpid(),
        start_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    ledger['stages'].append(stage);live={};completed_seconds=0.;begin=time.monotonic();failure=None;last_saved=[0.]

    def persist(force=False):
        used=completed_seconds+sum(time.monotonic()-info['start'] for info in live.values())
        stage['GPU_wall_hours_including_live']=used/3600
        stage['elapsed_seconds']=time.monotonic()-begin
        ledger['measured_GPU_wall_hours']=base+used/3600
        if force or time.monotonic()-last_saved[0]>=5:
            save(a.ledger,ledger);last_saved[0]=time.monotonic()
        return used

    try:
        while any(j['status']=='PENDING' for j in jobs) or live:
            used=persist()
            # Reserve termination grace for each of our own possible GPUs.
            if used>=budget*3600-10*len(gpus):failure='REGISTERED_GPU_WALL_LIMIT';break
            if time.monotonic()-begin>allocation['maximum_stage_wall_seconds']:failure='REGISTERED_STAGE_WALL_LIMIT';break
            for gpu,info in list(live.items()):
                job=info['job'];process=info['process'];code=process.poll()
                if code is None and time.monotonic()-info['start']>job['maximum_seconds']:
                    failure='JOB_TIMEOUT '+job['name'];break
                if code is not None:
                    seconds=time.monotonic()-info['start'];completed_seconds+=seconds;info['stream'].close();del live[gpu]
                    job.update(status='COMPLETE' if code==0 else 'FAILED_EVIDENCE_RETAINED',exit_code=code,wall_seconds=seconds)
                    print(json.dumps(dict(stage=stage['stage'],job=job['name'],status=job['status'],seconds=seconds)),flush=True)
                    if code!=0:failure='JOB_FAILED '+job['name'];break
            if failure:break
            for gpu in gpus:
                if gpu in live:continue
                pending=next((j for j in jobs if j['status']=='PENDING'),None)
                if pending is None:break
                memory=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
                if memory:continue
                path=Path(pending['log']);path.parent.mkdir(parents=True,exist_ok=True);stream=path.open('x')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                process=subprocess.Popen(pending['command'],cwd=source,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                live[gpu]=dict(process=process,stream=stream,job=pending,start=time.monotonic())
                pending.update(status='RUNNING',gpu=gpu,pid=process.pid)
                print(json.dumps(dict(stage=stage['stage'],job=pending['name'],status='RUNNING',gpu=gpu,pid=process.pid)),flush=True)
            persist()
            time.sleep(.5)
    except BaseException as error:
        failure=repr(error)
        raise
    finally:
        for gpu,info in list(live.items()):
            process=info['process']
            if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
        deadline=time.monotonic()+8
        for gpu,info in list(live.items()):
            process=info['process']
            try:process.wait(timeout=max(.01,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            elapsed=time.monotonic()-info['start'];completed_seconds+=elapsed;info['stream'].close()
            info['job'].update(status='STOPPED_OWNED_GROUP_EVIDENCE_RETAINED',exit_code=process.returncode,wall_seconds=elapsed)
            del live[gpu]
        stage.update(status='FAILED_EVIDENCE_RETAINED' if failure else 'COMPLETE',failure=failure)
        for job in jobs:
            if job['status']=='PENDING':job['status']='NOT_RUN_STAGE_STOP'
        persist(force=True)
    if failure:raise RuntimeError(failure)


if __name__=='__main__':main()
