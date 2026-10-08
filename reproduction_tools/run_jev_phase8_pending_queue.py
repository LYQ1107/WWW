"""Finish the sealed queue while keeping pre-existing scientific workers alive."""
import concurrent.futures,os,queue,subprocess,threading,time,json
from jev_phase8_common import *

def main():
    protect();start=binding();assert not start['worktree_dirty']
    original=json.loads((OUT/'bootstrap/V2_FORK_QUEUE_PLAN.json').read_text());q=queue.Queue();external=[];tasks=[]
    for t in original['tasks']:
        d=OUT/'corrective_forks_v2'/f'video{t["video"]:02d}'/f'shard{t["first"]:03d}_{t["last"]:03d}'
        if d.exists():external.append({'task':t,'directory':str(d),'status_at_handoff':'COMPLETE'if(d/'FORK_RESULT.json').exists()else'INFLIGHT_PRESERVED'})
        else:tasks.append(t);q.put(t)
    save(OUT/'bootstrap/PENDING_QUEUE_PLAN.json',{'binding':start,'tasks':tasks,'external_existing':external,'GPU_peak_reservation_MiB':12288,'scientific_selection_unchanged':True})
    runs=[];lock=threading.Lock();reserved={g:0 for g in(1,4,5,6,7)}
    def worker(gpu,slot):
        while True:
            try:t=q.get_nowait()
            except queue.Empty:return
            while True:
                with lock:
                    free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                    if free-reserved[gpu]>=12288:reserved[gpu]+=12288;break
                time.sleep(2)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            cmd=[PYTHON,str(ROOT/'reproduction_tools/run_jev_phase8_corrective_forks_v2.py'),'--video',str(t['video']),'--start-index',str(t['first']),'--stop-index',str(t['last'])]
            log=OUT/'bootstrap'/f'pending_forks{t["video"]:02d}_{t["first"]:03d}_{t["last"]:03d}.log'
            with log.open('w')as f:
                p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
                r={**t,'GPU':gpu,'slot':slot,'PID':p.pid,'command':cmd,'log':str(log),'start_time':time.time()}
                with lock:runs.append(r)
                code=p.wait()
                with lock:r.update(exit_code=code,end_time=time.time());reserved[gpu]-=12288
            print(json.dumps(r),flush=True)
            save(OUT/'bootstrap/PENDING_QUEUE_PROGRESS.json',{'status':'RUNNING','runs':runs})
            if code:raise RuntimeError('failed immutable native shard; do not overwrite')
    slots=[(4,0),(5,0),(6,0),(7,0),(1,0),(4,1),(5,1),(6,1),(7,1)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(slots))as pool:
        for f in concurrent.futures.as_completed([pool.submit(worker,g,s)for g,s in slots]):f.result()
    # The preserved earlier workers finish independently. All original
    # planned outputs, including their negative cases, must exist unchanged.
    while True:
        unfinished=[e for e in external if not(Path(e['directory'])/'FORK_RESULT.json').exists()]
        if not unfinished:break
        save(OUT/'bootstrap/PENDING_QUEUE_EXTERNAL_WAIT.json',{'status':'WAIT_EXISTING_NATIVE_WORKERS','unfinished':unfinished});time.sleep(5)
    save(OUT/'bootstrap/PENDING_QUEUE_EXECUTION.json',{'status':'COMPLETE_ALL_ORIGINAL_PLANNED_SHARDS','binding':start,'runs':runs,'external_existing':external});protect()
if __name__=='__main__':main()
