"""Bounded dependency-aware queue; all planned seeds/videos, no outcome selection."""
import os
import shutil
import subprocess
import time
from jev_phase14_common import *
from jev_phase14_queue import gpu_status

def main():
    protect();source=binding();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    assert read(REPORTS/'NATIVE_CACHE_PARITY.json')['status']=='PASS'
    protocol=read(REPORTS/'FORMAL_PROTOCOL.json');jobs=[]
    def add(key,stage,args,outputs,deps=(),formal=None):
        jobs.append(dict(key=key,stage=stage,args=args,outputs=[str(p) for p in outputs],deps=list(deps),formal=str(formal) if formal else None,status='PENDING'))
    for v in protocol['architectures']:
        for s in protocol['seeds']:
            formal=OUT/'formal_v2'/v/'availability_joint'/f'seed{s}/RESULT.json'
            deps=[]
            for video in TRAIN:
                key=f'collect_{v}_{s}_{video}';deps.append(key)
                out=OUT/'onpolicy_collect_v2'/v/f'seed{s}'/f'video{video:02d}/RESULT.json'
                add(key,'collect',['jev_phase14_onpolicy_collect.py','--variant',v,'--seed',str(s),'--video',str(video)],[out],formal=formal)
            for arm in ['onpolicy','oldloss_onpolicy','offpolicy']:
                root=OUT/f'{arm}_v2'/v/'availability_joint'/f'seed{s}'
                key=f'train_{arm}_{v}_{s}'
                add(key,'extra_train',['jev_phase14_onpolicy_train.py','--variant',v,'--seed',str(s),'--arm',arm],[root/'RESULT.json',root/'NOT_RUN.json'],deps,formal)
                for video in DEV:
                    out=OUT/f'{arm}_online_v2'/f'{v}_seed{s}'/f'video{video:02d}/RESULT.json'
                    add(f'eval_{arm}_{v}_{s}_{video}','extra_online',['jev_phase14_online.py','--variant',v,'--seed',str(s),'--phase',arm,'--video',str(video)],[out],[key],formal)
            for video in DEV:
                out=OUT/'formal_online_v2'/f'{v}_seed{s}'/f'video{video:02d}/RESULT.json'
                add(f'eval_formal_{v}_{s}_{video}','formal_online',['jev_phase14_online.py','--variant',v,'--seed',str(s),'--video',str(video)],[out],formal=formal)
                if s==20261009:
                    out=OUT/'formal_online_v2'/f'{v}_seed{s}_uncached'/f'video{video:02d}/RESULT.json'
                    add(f'parity_{v}_{s}_{video}','uncached_online',['jev_phase14_online.py','--variant',v,'--seed',str(s),'--video',str(video),'--uncached'],[out],formal=formal)
    priority={'collect':0,'extra_train':1,'formal_online':2,'extra_online':3,'uncached_online':4}
    jobs.sort(key=lambda j:priority[j['stage']]);index={j['key']:j for j in jobs};active={};queue=OUT/'completion_queue_v2';queue.mkdir(parents=True,exist_ok=True);begin=time.monotonic()
    for j in jobs:
        existing=next((Path(p) for p in j['outputs'] if Path(p).exists()),None)
        if existing:
            r=read(existing);j.update(status='DONE' if r['status']=='COMPLETE' else 'SKIPPED',result=str(existing),result_SHA256=sha(existing))
    while any(j['status'] in ['PENDING','RUNNING'] for j in jobs):
        for gpu,(process,handle,j) in list(active.items()):
            code=process.poll()
            if code is None:continue
            handle.close();existing=next((Path(p) for p in j['outputs'] if Path(p).exists()),None)
            if code==0 and existing:
                r=read(existing);j.update(status='DONE' if r['status']=='COMPLETE' else 'SKIPPED',result=str(existing),result_SHA256=sha(existing),returncode=code)
            else:j.update(status='FAILED',returncode=code)
            print('PHASE14_QUEUE_FINISHED',j['key'],j['status'],flush=True);del active[gpu]
        early_unfinished=any(j['stage'] in ['collect','extra_train'] and j['status'] in ['PENDING','RUNNING'] for j in jobs)
        ready=[]
        for j in jobs:
            if j['status']!='PENDING':continue
            parents=[index[d] for d in j['deps']]
            if any(d['status'] in ['FAILED','SKIPPED'] for d in parents):
                j.update(status='SKIPPED',reason='prerequisite failed or own-state scientific qualification did not pass',prerequisites=[dict(key=d['key'],status=d['status']) for d in parents]);continue
            if any(d['status']!='DONE' for d in parents):continue
            if j['formal'] and not Path(j['formal']).exists():continue
            if j['stage'].endswith('online') and early_unfinished:continue
            ready.append(j)
        candidates=[g for g in gpu_status() if g[0]!=9 and g[0] not in active and g[2]>=8192]
        candidates.sort(key=lambda g:(g[1]>1000,g[1],g[3],g[0]))
        for gpu,used,free,util in candidates:
            if not ready or len(active)>=8:break
            if shutil.disk_usage(OUT).free<3500000000:break
            j=ready.pop(0);log=queue/(j['key']+'.log');handle=log.open('a');env=os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
            cmd=[PYTHON,str(ROOT/'reproduction_tools'/j['args'][0])]+j['args'][1:]
            process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT);j.update(status='RUNNING',gpu=gpu,pid=process.pid,log=str(log),command=cmd,source_commit=source['source_commit']);active[gpu]=(process,handle,j)
            print('PHASE14_QUEUE_LAUNCH',gpu,process.pid,j['key'],flush=True)
        counts={status:sum(j['status']==status for j in jobs) for status in ['PENDING','RUNNING','DONE','SKIPPED','FAILED']}
        save(queue/'PROGRESS.json',dict(status='RUNNING',binding=source,total=len(jobs),counts=counts,active=[dict(key=j['key'],gpu=g,pid=p.pid,log=j['log']) for g,(p,h,j) in active.items()],disk_free_bytes=shutil.disk_usage(OUT).free,seconds=time.monotonic()-begin))
        save(queue/'JOBS.json',jobs);time.sleep(10)
    save(queue/'RESULT.json',dict(status='COMPLETE_WITH_FAILURES' if any(j['status']=='FAILED' for j in jobs) else 'COMPLETE',binding=source,jobs=jobs,seconds=time.monotonic()-begin))
    print('PHASE14_BOUND_EXECUTION_QUEUE_COMPLETE',len(jobs),flush=True)

if __name__=='__main__':main()
