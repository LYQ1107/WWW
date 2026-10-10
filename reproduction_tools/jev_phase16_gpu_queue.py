"""Bounded GPU queue that permits stacking when actual free VRAM allows it."""
import time
from jev_phase16_common import *
from jev_phase15_queue import gpus


def run_stack_queue(source,jobs,name,max_active=6):
    source=Path(source);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip()
    assert max_active<=read(REPORTS/'PREREGISTRATION.json')['budget']['max_GPU_workers']
    folder=OUT/name;folder.mkdir(parents=True,exist_ok=True)
    pending=list(jobs);active={};done=[];failed=[];launches=[];start=time.monotonic()
    # Additional conservative headroom is held even after CUDA allocation is
    # visible. An idle, process-free card takes priority; otherwise low GPU
    # utilization allows several workers on one card without exceeding six.
    reservation_MiB=4096;minimum_free_MiB=8192
    while pending or active:
        for pid,(gpu,process,stream,job) in list(active.items()):
            code=process.poll()
            if code is None:continue
            stream.close();item=dict(job,returncode=code,GPU=gpu)
            good=code==0 and Path(job['result']).exists() and read(job['result'])['status'] in ['COMPLETE','PASS']
            if good:assert read(job['result'])['binding']['source_commit']==head
            (done if good else failed).append(item);del active[pid]
            print('PHASE16_STACK_JOB_FINISHED',job['key'],code,flush=True)
        snapshot=gpus();assigned={g:sum(g==item[0] for item in active.values()) for g,*_ in snapshot}
        newly_assigned={g:0 for g,*_ in snapshot}
        while pending and len(active)<max_active:
            choices=[item for item in snapshot if item[2]-assigned[item[0]]*reservation_MiB>=minimum_free_MiB]
            if not choices:break
            choices.sort(key=lambda g:(g[4]+newly_assigned[g[0]]>0,
                min(100,g[3]+10*newly_assigned[g[0]]),g[1]+newly_assigned[g[0]]*reservation_MiB,g[4],g[0]))
            gpu,used,free,utilization,count=choices[0];job=pending.pop(0)
            if Path(job['result']).exists():
                r=read(job['result']);assert r['binding']['source_commit']==head and r['status'] in ['COMPLETE','PASS']
                done.append(dict(job,returncode=0,resumed=True));continue
            storage_guard();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',
                MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
            log=folder/(job['key']+'.log');stream=log.open('a')
            command=[PYTHON,str(source/'reproduction_tools'/job['script'])]+job['args']
            process=subprocess.Popen(command,cwd=source,env=env,stdout=stream,stderr=subprocess.STDOUT)
            active[process.pid]=(gpu,process,stream,job)
            launches.append(dict(key=job['key'],GPU=gpu,pid=process.pid,free_MiB_at_snapshot=free,
                other_and_current_processes_at_snapshot=count,utilization_at_snapshot=utilization,
                own_workers_on_card_before_launch=assigned[gpu],reserved_MiB_per_worker=reservation_MiB,
                source_commit=head,log=str(log),command=command))
            assigned[gpu]+=1;newly_assigned[gpu]+=1
            print('PHASE16_STACK_JOB_LAUNCH',gpu,process.pid,job['key'],flush=True)
        save(folder/'PROGRESS.json',dict(status='RUNNING',total=len(jobs),done=len(done),failed=len(failed),
            pending=len(pending),active=[dict(GPU=g,pid=pid,key=j['key']) for pid,(g,p,s,j) in active.items()],
            seconds=time.monotonic()-start))
        if pending or active:time.sleep(10)
    result=dict(status='COMPLETE' if not failed else 'COMPLETE_WITH_FAILURES',binding=binding(),
        actor_source_commit=head,done=done,failed=failed,launches=launches,seconds=time.monotonic()-start,
        min_free_MiB=minimum_free_MiB,reserved_MiB_per_worker=reservation_MiB,max_active=max_active)
    save(folder/'RESULT.json',result);save(folder/'PROGRESS.json',dict(status=result['status'],done=len(done),
        failed=len(failed),pending=0,active=[],total=len(jobs),result=ref(folder/'RESULT.json')))
    return result
