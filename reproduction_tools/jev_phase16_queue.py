"""Pinned sparse actors, free-VRAM scheduling and nonstale progress."""
import argparse
import time
from jev_phase16_common import *
from jev_phase15_queue import gpus

def pin(name):
    protect();storage_guard()
    destination=OUT/name;head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if destination.exists():
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=destination,text=True).strip()==head
        return destination
    subprocess.run(['git','worktree','add','--no-checkout','--detach',str(destination),head],cwd=ROOT,check=True)
    subprocess.run(['git','read-tree','HEAD'],cwd=destination,check=True)
    files=subprocess.check_output(['git','ls-files','-z'],cwd=destination).split(b'\0');keep=[];skip=[]
    for path in filter(None,files):
        p=path.decode();omit=p.startswith('TrackEval/data/') or p.startswith('reports/') and not p.startswith(
            ('reports/JEV_PHASE13/','reports/JEV_PHASE14/','reports/JEV_PHASE15/','reports/JEV_PHASE16/'))
        (skip if omit else keep).append(path)
    subprocess.run(['git','update-index','--skip-worktree','-z','--stdin'],input=b'\0'.join(skip)+b'\0',cwd=destination,check=True)
    subprocess.run(['git','checkout-index','-z','--stdin'],input=b'\0'.join(keep)+b'\0',cwd=destination,check=True)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=destination,text=True).strip()
    save(OUT/(name+'.json'),dict(status='PINNED',path=str(destination),commit=head,files=len(keep),omitted=len(skip)))
    return destination

def run_queue(source,jobs,name,max_active=6):
    source=Path(source);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip()
    folder=OUT/name;folder.mkdir(parents=True,exist_ok=True)
    pending=list(jobs);active={};done=[];failed=[];launches=[];start=time.monotonic()
    while pending or active:
        for gpu,(process,stream,job) in list(active.items()):
            code=process.poll()
            if code is None:continue
            stream.close();item=dict(job,returncode=code)
            good=code==0 and Path(job['result']).exists() and read(job['result'])['status'] in ['COMPLETE','PASS']
            (done if good else failed).append(item);del active[gpu]
            print('PHASE16_JOB_FINISHED',job['key'],code,flush=True)
        candidates=[g for g in gpus() if g[0] not in active and g[2]>=8192]
        candidates.sort(key=lambda g:(g[4]>0,g[4],g[3],g[1],g[0]))
        for gpu,used,free,utilization,process_count in candidates:
            if not pending or len(active)>=max_active:break
            storage_guard();job=pending.pop(0)
            if Path(job['result']).exists():
                r=read(job['result']);assert r['binding']['source_commit']==head and r['status'] in ['COMPLETE','PASS']
                done.append(dict(job,returncode=0,resumed=True));continue
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',
                MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
            log=folder/(job['key']+'.log');stream=log.open('a')
            command=[PYTHON,str(source/'reproduction_tools'/job['script'])]+job['args']
            process=subprocess.Popen(command,cwd=source,env=env,stdout=stream,stderr=subprocess.STDOUT)
            active[gpu]=(process,stream,job);launches.append(dict(key=job['key'],GPU=gpu,pid=process.pid,
                free_MiB_at_launch=free,compute_processes_at_launch=process_count,source_commit=head,log=str(log),command=command))
            print('PHASE16_JOB_LAUNCH',gpu,process.pid,job['key'],flush=True)
        save(folder/'PROGRESS.json',dict(status='RUNNING',total=len(jobs),done=len(done),failed=len(failed),
            pending=len(pending),active=[dict(GPU=g,pid=p.pid,key=j['key']) for g,(p,s,j) in active.items()],seconds=time.monotonic()-start))
        if pending or active:time.sleep(10)
    result=dict(status='COMPLETE' if not failed else 'COMPLETE_WITH_FAILURES',binding=binding(),actor_source_commit=head,
        done=done,failed=failed,launches=launches,seconds=time.monotonic()-start)
    save(folder/'RESULT.json',result);save(folder/'PROGRESS.json',dict(status=result['status'],done=len(done),
        failed=len(failed),pending=0,active=[],total=len(jobs),result=ref(folder/'RESULT.json')))
    return result

def main():
    protocol=read(REPORTS/'PREREGISTRATION.json');source=pin('source_P0_v1');jobs=[]
    for item in protocol['P0']['full_replays']:
        v,p=item['video'],item['policy'];jobs.append(dict(key=f'full_{p}_video{v}',script='jev_phase16_candidate_audit.py',
            args=['--video',str(v),'--policy',p,'--scope','full'],result=str(OUT/'P0/full'/p/f'video{v:02d}/RESULT.json')))
    for p in ['v1','v2','v3']:
        for v in TRAIN:
            jobs.append(dict(key=f'windows_{p}_video{v}',script='jev_phase16_candidate_audit.py',
                args=['--video',str(v),'--policy',p,'--scope','windows'],result=str(OUT/'P0/windows'/p/f'video{v:02d}/RESULT.json')))
    r=run_queue(source,jobs,'P0_queue_v1');assert not r['failed'],r['failed']

if __name__=='__main__':main()
