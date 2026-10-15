"""V2 four-GPU-hour watchdog, independent of the untouched V1 ledger."""
import argparse,fcntl
from gmt_jev_verifier_common import *
def update(name,fields):
    with (OUT/'REGISTRY.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);r=read(REPORTS/'EXPERIMENT_REGISTRY.json')
        r['jobs'].setdefault(name,{}).update(fields);save(REPORTS/'EXPERIMENT_REGISTRY.json',r)
def spent():
    return sum((max(0,time.time()-j['started_wall_time']) if j['status']=='RUNNING' else j.get('seconds',0))/3600 for j in read(REPORTS/'EXPERIMENT_REGISTRY.json')['jobs'].values() if j.get('GPU') is not None)
def main(name,gpu,script,args):
    reserve();assert name not in read(REPORTS/'EXPERIMENT_REGISTRY.json')['jobs'];assert spent()<4
    folder=OUT/'jobs'/name;folder.mkdir(parents=True,exist_ok=False);log=folder/'OUTPUT.log'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    command=[PYTHON,str(ROOT/'reproduction_tools'/script),*args];begin=time.monotonic()
    with log.open('w') as f:
        p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        update(name,dict(status='RUNNING',PID=p.pid,GPU=gpu,command=command,log=str(log),started_wall_time=time.time(),source=binding()))
        print('V2_STARTED',name,p.pid,gpu,flush=True);reason=None
        while p.poll() is None:
            jobs=read(REPORTS/'EXPERIMENT_REGISTRY.json')['jobs'].values()
            count=max(1,sum(j['status']=='RUNNING' and j.get('GPU') is not None for j in jobs))
            window=min(30,max(0,(4-spent())*3600/count-10))
            try:reserve()
            except AssertionError:reason='V2 storage/free-space limit'
            if window<=0:reason='V2 aggregate four-GPU-hour cap, ten-second exit reserve'
            if time.monotonic()-begin>2*3600:reason='V2 finite per-job two-hour limit'
            if reason:
                p.terminate()
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5)
                break
            try:p.wait(timeout=window)
            except subprocess.TimeoutExpired:pass
    code=p.wait();update(name,dict(status='COMPLETE' if code==0 else 'FAILED',exit_code=code,seconds=time.monotonic()-begin,log=ref(log),stop_reason=reason))
    print('V2_TERMINAL',name,code,'GPU_hours',spent(),flush=True)
    if code:raise SystemExit(code)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--script',required=True);a,rest=p.parse_known_args()
    main(a.name,a.gpu,a.script,rest[1:] if rest and rest[0]=='--' else rest)
