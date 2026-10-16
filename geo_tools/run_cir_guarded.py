"""Single owned GPU process group with registered cap and durable heartbeats."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ledger',type=Path,default=Path('results/CIR_GMT/RESOURCE_LEDGER.json'))
    p.add_argument('--stage',required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--seconds',type=int,required=True);p.add_argument('--log',type=Path,required=True)
    p.add_argument('--cwd',type=Path,default=Path.cwd());p.add_argument('--source-commit')
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();ledger=json.loads(a.ledger.read_text())
    if any(s['status'].startswith('RUNNING') for s in ledger['stages']):raise ValueError('Concurrent ledger writer not supported')
    if a.seconds<=0:raise ValueError('Positive bounded allocation required')
    total_cap=ledger.get('allocated_total_GPU_wall_hours',ledger['initial_allocation_GPU_wall_hours'])
    if ledger['measured_GPU_wall_hours']+a.seconds/3600>total_cap:raise ValueError('Registered CIR stage cap exceeded')
    used=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    if used:raise ValueError('Choose an idle GPU; no preemption authorized')
    cmd=a.command[1:] if a.command[:1]==['--'] else a.command
    if not cmd:raise ValueError('Missing actual command')
    if a.source_commit:
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.cwd,text=True).strip()!=a.source_commit or subprocess.check_output(['git','status','--porcelain'],cwd=a.cwd):
            raise ValueError('Exact clean source snapshot required')
    a.log.parent.mkdir(parents=True,exist_ok=True)
    if a.log.exists():raise FileExistsError(a.log)
    stage=dict(stage=a.stage,status='RUNNING',device_physical_index=a.gpu,maximum_process_seconds=a.seconds,
        command=cmd,log=str(a.log.resolve()),start_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        supervisor_pid=os.getpid())
    stage.update(source_worktree=str(a.cwd.resolve()),source_commit=a.source_commit)
    ledger['stages'].append(stage);save(a.ledger,ledger)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    t=time.monotonic();code=None
    with a.log.open('x') as stream:
        process=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True,cwd=a.cwd)
        stage['child_pid']=process.pid
        try:
            while True:
                try:code=process.wait(timeout=min(15,max(.01,a.seconds-(time.monotonic()-t))));break
                except subprocess.TimeoutExpired:
                    stage['elapsed_seconds']=time.monotonic()-t;save(a.ledger,ledger)
                    if time.monotonic()-t>=a.seconds:
                        stage['stop_reason']='REGISTERED_WALL_LIMIT';break
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                code=124
            elapsed=time.monotonic()-t
            stage.update(status='COMPLETE' if code==0 else 'FAILED_EVIDENCE_RETAINED',exit_code=code,
                         wall_seconds=elapsed,conservative_GPU_wall_hours=elapsed/3600)
            ledger['measured_GPU_wall_hours']+=elapsed/3600;save(a.ledger,ledger)
    print(json.dumps(stage),flush=True)
    raise SystemExit(code)


if __name__=='__main__':main()
