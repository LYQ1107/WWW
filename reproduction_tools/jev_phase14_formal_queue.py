"""Launch only preregistered pilot-qualified equal-budget three-seed models."""
import os
import subprocess
import time
from jev_phase14_common import *
from jev_phase14_queue import gpu_status

def main():
    protect();protocol=read(REPORTS/'FORMAL_PROTOCOL.json');source=OUT/'source_pilot_v2'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()==protocol['source_training_pin']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip()
    pending=[(v,s) for v in protocol['architectures'] for s in protocol['seeds']]
    active={};done=[];failed=[];queue=OUT/'formal_queue_v2';queue.mkdir(parents=True,exist_ok=True)
    version=protocol['execution_version'];begin=time.monotonic();launches=[]
    while pending or active:
        for gpu,(p,h,variant,seed) in list(active.items()):
            code=p.poll()
            if code is None:continue
            h.close();case=OUT/f'formal_v{version}'/variant/protocol['objective']/f'seed{seed}'
            item=dict(variant=variant,seed=seed,returncode=code,result=str(case/'RESULT.json'))
            (done if code==0 and (case/'RESULT.json').exists() else failed).append(item)
            del active[gpu]
        # Keep one empty V100 for identical-hardware frontend/native profiling.
        available=[g for g in gpu_status() if g[0] not in active and g[0]!=9 and g[1]<1000 and g[2]>=8192]
        available.sort(key=lambda g:g[0])
        for gpu,used,free,util in available:
            if not pending or len(active)>=8:break
            variant,seed=pending.pop(0);case=OUT/f'formal_v{version}'/variant/protocol['objective']/f'seed{seed}'
            if (case/'RESULT.json').exists():done.append(dict(variant=variant,seed=seed,result=str(case/'RESULT.json'),returncode=0));continue
            if (case/'FAILED.json').exists():failed.append(dict(variant=variant,seed=seed,failure=str(case/'FAILED.json')));continue
            log=queue/f'{variant}_seed{seed}.log';h=log.open('a');env=os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
            cmd=[PYTHON,str(source/'reproduction_tools/jev_phase14_train.py'),'--phase','formal','--variant',variant,'--loss',protocol['objective'],'--seed',str(seed)]
            p=subprocess.Popen(cmd,cwd=source,env=env,stdout=h,stderr=subprocess.STDOUT);active[gpu]=(p,h,variant,seed)
            launches.append(dict(gpu=gpu,pid=p.pid,command=cmd,source_commit=protocol['source_training_pin'],formal_protocol_SHA256=sha(REPORTS/'FORMAL_PROTOCOL.json')))
            print('FORMAL_LAUNCH',gpu,p.pid,variant,seed,flush=True)
        save(queue/'PROGRESS.json',dict(status='RUNNING',total=9,completed=len(done),failed=len(failed),pending=len(pending),
            active=[dict(gpu=g,pid=p.pid,variant=v,seed=s,progress=read(OUT/f'formal_v{version}'/v/protocol['objective']/f'seed{s}/PROGRESS.json') if (OUT/f'formal_v{version}'/v/protocol['objective']/f'seed{s}/PROGRESS.json').exists() else {'status':'STARTING'}) for g,(p,h,v,s) in active.items()],seconds=time.monotonic()-begin))
        time.sleep(10)
    save(queue/'RESULT.json',dict(status='PASS' if not failed else 'COMPLETE_WITH_FAILURES',binding=binding(),done=done,failed=failed,launches=launches,seconds=time.monotonic()-begin))
    save(queue/'PROGRESS.json',dict(status='COMPLETE',completed=len(done),failed=len(failed),total=9))
    print('PHASE14_FORMAL_QUEUE_COMPLETE',len(done),len(failed),flush=True)

if __name__=='__main__':main()
