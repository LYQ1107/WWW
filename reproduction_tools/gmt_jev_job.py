"""Register finite jobs, enforce destination/storage/time budgets, retain failures."""
import argparse,fcntl
from gmt_jev_common import *
def update(name,fields):
 with (OUT/'REGISTRY.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);r=read(REPORTS/'EXPERIMENT_REGISTRY.json');r['jobs'].setdefault(name,{}).update(fields);save(REPORTS/'EXPERIMENT_REGISTRY.json',r)
def spent():
 now=time.time();r=read(REPORTS/'EXPERIMENT_REGISTRY.json')
 return sum((d.get('seconds',0) if d['status']!='RUNNING' else max(0,now-d.get('started_wall_time',now)))/3600 for d in r['jobs'].values() if d.get('GPU') is not None)
def main(name,gpu,script,args):
 reserve();r=read(REPORTS/'EXPERIMENT_REGISTRY.json');assert name not in r['jobs']
 used=spent();assert used<12
 folder=OUT/'jobs'/name;folder.mkdir(parents=True,exist_ok=False);log=folder/'OUTPUT.log'
 env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
 command=[PYTHON,str(ROOT/'reproduction_tools'/script),*args];begin=time.monotonic()
 with log.open('w') as f:
  p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
  update(name,dict(status='RUNNING',PID=p.pid,GPU=gpu,command=command,log=str(log),source=binding(),started_wall_time=time.time()))
  print('STARTED',name,p.pid,gpu,flush=True)
  while True:
   try:code=p.wait(timeout=60);break
   except subprocess.TimeoutExpired:
    try:reserve();assert spent()<12 and time.monotonic()-begin<3*3600
    except AssertionError:p.terminate();code=p.wait(timeout=30);break
 update(name,dict(status='COMPLETE' if code==0 else 'FAILED',exit_code=code,seconds=time.monotonic()-begin,log=ref(log)))
 r=read(REPORTS/'EXPERIMENT_REGISTRY.json')
 print('TERMINAL',name,r['jobs'][name]['status'],r['jobs'][name]['seconds'],flush=True)
 if code:raise SystemExit(code)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--gpu',type=int);p.add_argument('--script',required=True);a,rest=p.parse_known_args()
 # An explicit separator preserves child flags such as its own --gpu.
 if rest and rest[0]=='--':rest=rest[1:]
 main(a.name,a.gpu,a.script,rest)
