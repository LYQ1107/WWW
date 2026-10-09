"""Prepare fixed external data, wait for frozen models, evaluate every planned case."""
import os
import subprocess
import time
from jev_phase14_common import *
from jev_phase14_external_prepare import EXTERNAL,DATA
from jev_phase14_queue import gpu_status

def execute(script,args,log,gpu=None):
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    with log.open('a') as h:subprocess.run([PYTHON,str(ROOT/'reproduction_tools'/script)]+args,cwd=ROOT,env=env,stdout=h,stderr=subprocess.STDOUT,check=True)

def main():
    protect();source=binding();root=EXTERNAL/'completion_queue_v1';root.mkdir(parents=True,exist_ok=True)
    while not (DATA/'RESULT.json').exists():
        save(root/'PROGRESS.json',dict(status='WAITING_FOR_FIXED_DOWNLOAD',binding=source,download_progress=read(DATA/'PROGRESS.json') if (DATA/'PROGRESS.json').exists() else None));time.sleep(10)
    if not (EXTERNAL/'DATA_MANIFEST.json').exists():execute('jev_phase14_external_prepare.py',['--mode','prepare'],root/'prepare.log')
    if not (EXTERNAL/'stage1_cache_v1/RESULT.json').exists():
        available=[g for g in gpu_status() if g[0]!=9 and g[2]>=8192];available.sort(key=lambda g:(g[1]>1000,g[3],g[1],g[0]));assert available
        gpu=available[0][0];save(root/'PROGRESS.json',dict(status='REAL_FROZEN_FRONTEND',GPU=gpu,binding=source))
        execute('jev_phase14_external_prepare.py',['--mode','cache'],root/'frontend.log',gpu)
    protocol=read(REPORTS/'FORMAL_PROTOCOL.json');pending=[(v,s,True) for v in ['full','fixed_question','set_transformer'] for s in protocol['seeds']]
    pending += [(v,s,False) for v in protocol['architectures'] for s in protocol['seeds']]+[('cosine',20261009,False)]
    active={};done=[];failed=[]
    while pending or active:
        for gpu,(process,handle,case,log) in list(active.items()):
            code=process.poll()
            if code is None:continue
            handle.close();v,s,historical=case;name=('phase13_' if historical else 'phase14_')+f'{v}_seed{s}';path=EXTERNAL/'native_online_v1'/name/'RESULT.json';item=dict(variant=v,seed=s,historical=historical,returncode=code,log=str(log),result=str(path))
            (done if code==0 and path.exists() else failed).append(item);del active[gpu]
        available=[g for g in gpu_status() if g[0]!=9 and g[0] not in active and g[2]>=8192];available.sort(key=lambda g:(g[1]>1000,g[1],g[3],g[0]))
        # External choices were frozen before outcomes, nevertheless complete the
        # primary online queue before interpreting independent transfer scores.
        primary=OUT/'completion_queue_v2/RESULT.json';ready=primary.exists()
        for gpu,used,free,util in available:
            if not ready or not pending or len(active)>=4:break
            v,s,historical=pending.pop(0)
            if not historical and v!='cosine':assert (OUT/'formal_v2'/v/'availability_joint'/f'seed{s}/RESULT.json').exists()
            name=('phase13_' if historical else 'phase14_')+f'{v}_seed{s}';path=EXTERNAL/'native_online_v1'/name/'RESULT.json'
            if path.exists():done.append(dict(variant=v,seed=s,historical=historical,result=str(path)));continue
            log=root/(name+'.log');handle=log.open('a');args=['--variant',v,'--seed',str(s)]+(['--historical'] if historical else [])
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1');cmd=[PYTHON,str(ROOT/'reproduction_tools/jev_phase14_external_online.py')]+args
            p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT);active[gpu]=(p,handle,(v,s,historical),log);print('PHASE14_EXTERNAL_QUEUE_LAUNCH',gpu,name,flush=True)
        save(root/'PROGRESS.json',dict(status='WAITING_FOR_PRIMARY_ONLINE' if not ready else 'EVALUATING_FROZEN_EXTERNAL',binding=source,total=19,done=len(done),failed=len(failed),pending=len(pending),active=[dict(gpu=g,pid=p.pid,case=c,log=str(l)) for g,(p,h,c,l) in active.items()]));time.sleep(10)
    save(root/'RESULT.json',dict(status='COMPLETE_WITH_FAILURES' if failed else 'COMPLETE',binding=source,done=done,failed=failed))
    print('PHASE14_EXTERNAL_ALL_PLANNED_COMPLETE',len(done),len(failed),flush=True)

if __name__=='__main__':main()
