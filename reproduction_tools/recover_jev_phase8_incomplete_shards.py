"""Recover interrupted original shards without overwriting any prior artifact."""
import json,os,concurrent.futures,queue,subprocess,time
from jev_phase8_common import *

def main():
    protect();start=binding();assert not start['worktree_dirty'];q=queue.Queue();tasks=[]
    external=json.loads((OUT/'bootstrap/PENDING_QUEUE_PLAN.json').read_text())['external_existing']
    active=[]
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                if p.stat().st_uid==os.getuid():active.append((p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
            except (OSError,PermissionError):pass
    for item in external:
        d=Path(item['directory']);t=item['task']
        if(d/'FORK_RESULT.json').exists():continue
        signature=f'--video {t["video"]} --start-index {t["first"]} --stop-index {t["last"]}'
        assert not any(signature in a and'run_jev_phase8_corrective_forks_v2.py'in a for a in active),'do not duplicate active scientific worker'
        tasks.append(t);q.put(t)
    save(OUT/'bootstrap/RECOVERY_PLAN.json',{'status':'SEALED_RECOVER_ONLY_UNFINISHED_CASES','binding':start,'tasks':tasks,
        'cause':'scheduler exit / terminal process-group cleanup; original partial cases and every negative result preserved','overwrite_previous_files':False})
    runs=[]
    def worker(gpu):
        while True:
            try:t=q.get_nowait()
            except queue.Empty:return
            cmd=[PYTHON,str(ROOT/'reproduction_tools/run_jev_phase8_corrective_recovery.py'),'--video',str(t['video']),'--start-index',str(t['first']),'--stop-index',str(t['last'])]
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            log=OUT/'bootstrap'/f'recovery_{t["video"]:02d}_{t["first"]:03d}_{t["last"]:03d}.log'
            with log.open('w')as f:
                p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                r={**t,'GPU':gpu,'PID':p.pid,'command':cmd,'log':str(log),'start_time':time.time()};runs.append(r);r['exit_code']=p.wait();r['end_time']=time.time()
            assert r['exit_code']==0,'failed recovery retained; never overwrite'
            old=OUT/'corrective_forks_v2'/f'video{t["video"]:02d}'/f'shard{t["first"]:03d}_{t["last"]:03d}'
            new=OUT/'corrective_forks_recovery_v1'/f'video{t["video"]:02d}'/old.name/'FORK_RESULT.json'
            partial=old/'FORK_RESULT.partial.json';a=json.loads(partial.read_text()).get('events',[])if partial.exists()else[];b=json.loads(new.read_text())['events']
            result={'status':'COMPLETE_COMPOSED_FROM_IMMUTABLE_NATIVE_EVIDENCE','binding':start,'events':a+b,
                'source_results':[{'path':str(x),'sha256':sha(x)}for x in [partial,new]if x.exists()],
                'recovery_reason':'terminal cleanup interrupted the original worker; completed cases retained without outcome filtering'}
            assert not(old/'FORK_RESULT.json').exists();save(old/'FORK_RESULT.json',result)
            print(json.dumps(r),flush=True);save(OUT/'bootstrap/RECOVERY_PROGRESS.json',{'status':'RUNNING','runs':runs})
    with concurrent.futures.ThreadPoolExecutor(max_workers=5)as pool:
        for f in concurrent.futures.as_completed([pool.submit(worker,g)for g in(4,5,6,7,1)]):f.result()
    save(OUT/'bootstrap/RECOVERY_EXECUTION.json',{'status':'COMPLETE','binding':start,'runs':runs});protect()
if __name__=='__main__':main()
