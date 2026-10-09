"""Complete all preregistered native paired branches and matched TRAIN risk audits."""
import os,time,subprocess
from jev_phase14_common import *
from jev_phase14_queue import gpu_status

def env(gpu):
    e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1');return e

def main():
    protect();source=binding();assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()
    root=OUT/'diagnostic_queue_v1';root.mkdir(exist_ok=True)
    if not (OUT/'paired_native_future_v1/PREFIX_MANIFEST.json').exists():
        with (root/'prefix_capture.log').open('a') as h:
            subprocess.run([PYTHON,str(ROOT/'reproduction_tools/jev_phase14_paired_future.py'),'--mode','capture'],cwd=ROOT,env=env(9),stdout=h,stderr=subprocess.STDOUT,check=True)
    p=read(REPORTS/'FORMAL_PROTOCOL.json');jobs=[]
    for s in p['seeds']:
        jobs.append(dict(key=f'paired_{s}',args=['jev_phase14_paired_future.py','--mode','branches','--seed',str(s)],result=str(OUT/'paired_native_future_v1'/f'seed{s}/RESULT.json')))
    for phase in ['formal','onpolicy','oldloss_onpolicy','offpolicy']:
        for v in p['architectures']:
            for s in p['seeds']:
                jobs.append(dict(key=f'risk_{phase}_{v}_{s}',args=['jev_phase14_train_risk.py','--phase',phase,'--variant',v,'--seed',str(s)],result=str(OUT/'matched_TRAIN_native_risk_v1'/phase/f'{v}_seed{s}/RESULT.json')))
    active={};done=[];failed=[]
    while jobs or active:
        for gpu,(proc,h,j) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            h.close();j['returncode']=code;(done if code==0 and Path(j['result']).exists() else failed).append(j);del active[gpu]
            print('PHASE14_DIAGNOSTIC_FINISHED',j['key'],code,flush=True)
        candidates=[g for g in gpu_status() if g[0] not in active and g[2]>=8192];candidates.sort(key=lambda g:(g[1]>1000,g[3],g[1],g[0]))
        for gpu,used,free,util in candidates:
            if not jobs or len(active)>=4:break
            j=jobs.pop(0)
            if Path(j['result']).exists():done.append(j);continue
            log=root/(j['key']+'.log');h=log.open('a');j['log']=str(log);j['GPU']=gpu
            proc=subprocess.Popen([PYTHON,str(ROOT/'reproduction_tools'/j['args'][0])]+j['args'][1:],cwd=ROOT,env=env(gpu),stdout=h,stderr=subprocess.STDOUT);active[gpu]=(proc,h,j)
            print('PHASE14_DIAGNOSTIC_LAUNCH',gpu,j['key'],flush=True)
        save(root/'PROGRESS.json',dict(binding=source,total=39,done=len(done),failed=len(failed),pending=len(jobs),active=[dict(GPU=g,pid=v[0].pid,key=v[2]['key']) for g,v in active.items()]));time.sleep(10)
    save(root/'RESULT.json',dict(status='COMPLETE_WITH_FAILURES' if failed else 'COMPLETE',binding=source,done=done,failed=failed))

if __name__=='__main__':main()
