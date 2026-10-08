"""Four GPU worker processes; isolated outputs, persistent logs, no model search."""
import argparse,concurrent.futures,json,os,subprocess,time
from jev_phase7_common import *

def main(stage):
    jobs=[]
    if stage=='primary':
        for condition in ('B2','GMT','FIXED','DYNAMIC','MLP','BINARY'):
            for video in (9,10,11):jobs.append((condition,video,condition,[]))
        for video in (9,10,11):jobs.append(('B2',video,'B2_VALIDATION_ONLY',['--no-global']))
    elif stage=='controls':
        for condition in ('FIXED','DYNAMIC','MLP','BINARY','B2'):
            for video in (9,10,11):jobs.append((condition,video,condition+'_LEGACY_VALIDATOR',['--validator','LEGACY']))
        for video in (9,10,11):jobs.append(('B2',video,'B2_ORIGINAL_LEGACY',['--validator','LEGACY','--no-global']))
    else:
        for condition in ('FIXED','DYNAMIC','MLP','BINARY','B2'):
            for video in (9,10,11):
                base=json.loads((OUT/'closed_loop'/f'video{video:02d}/B2/result.json').read_text())
                assert base['complete_video']
                jobs.append((condition,video,condition+'_BUDGET',['--budget',str(base['budget_achieved'])]))
    protocol=ROOT/'docs/PHASE7_FAIR_BASELINE_PROTOCOL.md'
    plan={'stage':stage,'binding':binding(),'protocol_sha256':sha(protocol),'jobs':jobs,
          'gpu_priority':[4,5,6,7],'raw_output_root':str(OUT),'official_test_read':False}
    save(REPORTS/f'EXECUTION_PLAN_{stage.upper()}.json',plan)
    remaining=list(jobs);results=[]
    def worker(gpu):
        items=[]
        while True:
            with lock:
                if not remaining:break
                condition,video,tag,extra=remaining.pop(0)
            path=OUT/'logs'/f'{stage}_video{video:02d}_{tag}.log';path.parent.mkdir(parents=True,exist_ok=True)
            command=[PYTHON,'-u',str(ROOT/'reproduction_tools/run_jev_phase7_attribution.py'),
                     '--condition',condition,'--video',str(video),'--tag',tag]+extra
            if tag=='B2':command+=['--capture']
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            start=time.time()
            with path.open('w') as log:
                child=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            item={'condition':condition,'video':video,'tag':tag,'gpu':gpu,'exit_code':child.returncode,
                  'log':str(path),'log_sha256':sha(path),'command':command,'elapsed_seconds':time.time()-start}
            with lock:
                results.append(item);save(OUT/f'QUEUE_{stage.upper()}.json',{'plan':plan,'completed':results,'remaining':remaining})
                if child.returncode:
                    remaining.clear();print(json.dumps({'status':'FAILED','job':item}),flush=True)
            items.append(item)
            if child.returncode:break
        return items
    import threading
    lock=threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(worker,gpu)for gpu in (4,5,6,7)]
        for future in futures:future.result()
    save(REPORTS/f'EXECUTION_RESULTS_{stage.upper()}.json',{'plan':plan,'completed':results,
         'status':'COMPLETE'if len(results)==len(jobs) and all(r['exit_code']==0 for r in results)else 'FAILED'})
    assert len(results)==len(jobs) and all(r['exit_code']==0 for r in results)
    print(json.dumps({'stage':stage,'status':'COMPLETE','runs':len(results)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['primary','controls','budget']);main(p.parse_args().stage)
