"""Two bounded own experiment workers per GPU; immutable disjoint shards."""
import concurrent.futures,json,os,queue,subprocess,time
from pathlib import Path
from jev_phase8_common import *

def main():
    protect();start=binding();assert not start['worktree_dirty']
    q=queue.Queue();tasks=[]
    for v in(17,19,12,14,16,13,18):
        scan=json.loads((OUT/'opportunity_scan_v1'/f'video{v:02d}'/'SCAN_RESULT.json').read_text())
        prior=OUT/'corrective_forks_v1'/f'video{v:02d}'
        p=prior/('FORK_RESULT.json'if (prior/'FORK_RESULT.json').exists()else'FORK_RESULT.partial.json')
        old=json.loads(p.read_text()).get('events',[])if p.exists()else[]
        done={tuple(e['key'])+(e['row'],)for e in old}
        for first in range(0,len(scan['bounded_snapshots']),2):
            last=min(first+2,len(scan['bounded_snapshots']))
            records=scan['bounded_snapshots'][first:last]
            missing=[tuple(r['key'])+(row,)for r in records for row in r['selected_rows']if tuple(r['key'])+(row,)not in done]
            if missing:tasks.append({'video':v,'first':first,'last':last,'missing_event_keys':[list(k)for k in missing]})
    plan=OUT/'bootstrap'/'V2_FORK_QUEUE_PLAN.json';assert not plan.exists()
    save(plan,{'status':'SEALED_BEFORE_REMAINING_FORKS','binding':start,'GPUs':[4,5,6,7],'workers_per_GPU':2,'tasks':tasks,
        'selection_rule':'all uncompleted original representatives; prior COMPLETE cases reused regardless of sign; no utility filtering',
        'state_digest':'CONTROL/KEEP every payload; other interventions current/H8/H16/H32; all IDs every payload'})
    for t in tasks:q.put(t)
    runs=[]
    def worker(gpu,slot):
        while True:
            try:t=q.get_nowait()
            except queue.Empty:return
            v=t['video'];a=t['first'];b=t['last'];env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            command=[PYTHON,str(ROOT/'reproduction_tools/run_jev_phase8_corrective_forks_v2.py'),'--video',str(v),'--start-index',str(a),'--stop-index',str(b)]
            log=OUT/'bootstrap'/f'v2_forks{v:02d}_{a:03d}_{b:03d}.log'
            with log.open('w')as f:
                p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
                r={**t,'GPU':gpu,'slot':slot,'PID':p.pid,'command':command,'log':str(log),'start_time':time.time()};runs.append(r)
                r['exit_code']=p.wait();r['end_time']=time.time()
            save(OUT/'bootstrap'/'V2_FORK_QUEUE_PROGRESS.json',{'status':'RUNNING','runs':runs,'remaining_unlaunched_shards':q.qsize()})
            print(json.dumps(r),flush=True)
            if r['exit_code']:raise RuntimeError(f'immutable failed shard {v}/{a}/{b}; preserve evidence')
            q.task_done()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:
        for f in concurrent.futures.as_completed([pool.submit(worker,g,s)for g in(4,5,6,7)for s in(0,1)]):f.result()
    save(OUT/'bootstrap'/'V2_FORK_QUEUE_EXECUTION.json',{'status':'COMPLETE','binding':start,'runs':runs});protect()
if __name__=='__main__':main()
