"""Complete-video preregistered normalization controls and paper ByteTrack cascade."""
import concurrent.futures,os,subprocess
from jev_phase7_common import *

def worker(video):
    gpu=4+(video-9);env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    budget=json.loads((OUT/'closed_loop'/f'video{video:02d}'/'B2/result.json').read_text())['mechanism_counts']['reassociate_rows']
    jobs=[['run_jev_phase7_bytetrack.py','--video',str(video),'--threshold','.6'],
          ['run_jev_phase7_normalized.py','--video',str(video),'--validator','LEGACY'],
          ['run_jev_phase7_normalized.py','--video',str(video),'--budget',str(budget)]]
    results=[]
    for idx,args in enumerate(jobs):
        log=OUT/'logs'/f'supplement_video{video:02d}_{idx}.log'
        with log.open('w')as stream:
            code=subprocess.call([str(PYTHON),str(ROOT/'reproduction_tools'/args[0]),*args[1:]],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        results.append({'video':video,'gpu':gpu,'arguments':args,'log':str(log),'exit_code':code})
        assert code==0,(video,args,code)
    return results

if __name__=='__main__':
    protect();save(REPORTS/'EXECUTION_PLAN_SUPPLEMENTS.json',{'binding':binding(),'videos':[9,10,11],
       'normalization_preregistration_sha256':sha(ROOT/'docs/PHASE7_NORMALIZATION_CONTROL_PREREGISTRATION.md'),
       'bytetrack_preregistration_sha256':sha(ROOT/'docs/JEV_PHASE7_BYTETRACK_PAPER_THRESHOLD_PREREGISTRATION.md')})
    with concurrent.futures.ThreadPoolExecutor(max_workers=3)as pool:results=list(pool.map(worker,[9,10,11]))
    save(REPORTS/'EXECUTION_RESULTS_SUPPLEMENTS.json',{'status':'COMPLETE','results':results,'binding':binding()});protect()
