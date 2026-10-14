"""Register finite jobs, enforce destination/storage/time budgets, retain failures."""
import argparse
from gmt_jev_common import *
def main(name,gpu,script,args):
 reserve();r=read(REPORTS/'EXPERIMENT_REGISTRY.json');assert name not in r['jobs']
 used=sum(d.get('seconds',0)/3600 for d in r['jobs'].values() if d.get('GPU') is not None);assert used<12
 folder=OUT/'jobs'/name;folder.mkdir(parents=True,exist_ok=False);log=folder/'OUTPUT.log'
 env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
 command=[PYTHON,str(ROOT/'reproduction_tools'/script),*args];begin=time.monotonic()
 with log.open('w') as f:
  p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
  r['jobs'][name]=dict(status='RUNNING',PID=p.pid,GPU=gpu,command=command,log=str(log),source=binding());save(REPORTS/'EXPERIMENT_REGISTRY.json',r)
  print('STARTED',name,p.pid,gpu,flush=True)
  try:code=p.wait(timeout=min(3*3600,(12-used)*3600))
  except subprocess.TimeoutExpired:p.terminate();code=p.wait(timeout=30)
 r=read(REPORTS/'EXPERIMENT_REGISTRY.json');r['jobs'][name].update(status='COMPLETE' if code==0 else 'FAILED',exit_code=code,seconds=time.monotonic()-begin,log=ref(log));save(REPORTS/'EXPERIMENT_REGISTRY.json',r)
 print('TERMINAL',name,r['jobs'][name]['status'],r['jobs'][name]['seconds'],flush=True)
 if code:raise SystemExit(code)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--gpu',type=int);p.add_argument('--script',required=True);a,rest=p.parse_known_args();main(a.name,a.gpu,a.script,rest)
