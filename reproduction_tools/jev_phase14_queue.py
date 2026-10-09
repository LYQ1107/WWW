"""Bounded pilot queue, empty GPUs first; per-worker progress and errors retained."""
import argparse
import itertools
import os
import subprocess
import time
from jev_phase14_common import *

def gpu_status():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return [tuple(map(int,line.split(','))) for line in text.strip().splitlines()]

def main():
    protect();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    assert read(REPORTS/'ACTION_SEMANTICS_TESTS.json')['status']=='PASS'
    assert read(REPORTS/'DYNAMIC_QUESTION_DIAGNOSTICS.json')['status']=='COMPLETE'
    protocol=read(REPORTS/'PILOT_PROTOCOL.json');cases=list(itertools.product(protocol['models'],protocol['objectives']))
    queue=OUT/'pilot_queue_v1';queue.mkdir(parents=True,exist_ok=True)
    pending=[];done=[];failed=[];active={};started=time.monotonic()
    for variant,loss in cases:
        case=OUT/'pilot_v1'/variant/loss/'seed20261009'
        if (case/'RESULT.json').exists():done.append(dict(variant=variant,loss=loss,result=read(case/'RESULT.json')))
        elif (case/'FAILED.json').exists():failed.append(dict(variant=variant,loss=loss,failure=read(case/'FAILED.json')))
        else:pending.append((variant,loss))
    while pending or active:
        for gpu,(proc,handle,variant,loss) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            handle.close();case=OUT/'pilot_v1'/variant/loss/'seed20261009'
            if code==0 and (case/'RESULT.json').exists():done.append(dict(variant=variant,loss=loss,result=read(case/'RESULT.json')))
            else:failed.append(dict(variant=variant,loss=loss,returncode=code,log=str(queue/f'{variant}_{loss}.log')))
            del active[gpu]
        # Source worktree/policy are immutable while workers run. Resource
        # occupancy belongs to this queue only for its own child processes.
        candidates=[g for g in gpu_status() if g[0] not in active and g[2]>=8192 and g[3]<25]
        candidates.sort(key=lambda g:(g[1]>0,g[1],g[0]))
        for gpu,used,free,util in candidates:
            if not pending or len(active)>=9:break
            # Empty devices are sufficient here; stacking permitted by user but
            # leave unrelated occupied devices available unless no empty slots.
            if used>1000 and any(g[1]<=1000 for g in candidates):continue
            variant,loss=pending.pop(0);log=queue/f'{variant}_{loss}.log';handle=log.open('a')
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
            command=[PYTHON,str(ROOT/'reproduction_tools/jev_phase14_train.py'),'--variant',variant,'--loss',loss]
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
            active[gpu]=(proc,handle,variant,loss)
            print('PILOT_LAUNCH',gpu,proc.pid,variant,loss,flush=True)
        progress=dict(status='RUNNING',binding=binding(),total=len(cases),completed=len(done),failed=len(failed),pending=len(pending),
                      active=[dict(gpu=g,pid=p.pid,variant=v,loss=l,progress=read(OUT/'pilot_v1'/v/l/'seed20261009/PROGRESS.json') if (OUT/'pilot_v1'/v/l/'seed20261009/PROGRESS.json').exists() else {'status':'STARTING'}) for g,(p,h,v,l) in active.items()],seconds=time.monotonic()-started)
        save(queue/'PROGRESS.json',progress);time.sleep(10)
    save(queue/'RESULT.json',dict(status='PASS' if not failed else 'COMPLETE_WITH_FAILURES',binding=binding(),done=done,failed=failed,total=len(cases),seconds=time.monotonic()-started))
    save(queue/'PROGRESS.json',dict(status='COMPLETE',completed=len(done),failed=len(failed),total=len(cases)))
    print('PHASE14_PILOT_QUEUE_COMPLETE',len(done),len(failed),flush=True)

if __name__=='__main__':main()
