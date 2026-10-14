"""Registered subprocess launcher with bounded elapsed GPU-hours and disk guard."""
import argparse,fcntl,shutil
from current_research_goal import redirect_if_superseded
if redirect_if_superseded('reports/JEV_FASTCHECK/GOAL.json'):
 raise SystemExit(0)
from jev_fastcheck_common import *
def update(name,value):
 REPORTS.mkdir(parents=True,exist_ok=True)
 with (OUT/'REGISTRY.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);p=REPORTS/'EXPERIMENT_REGISTRY.json';d=read(p);d['jobs'][name]=value;save(p,d)
def main(name,gpu,script,args):
 OUT.mkdir(parents=True,exist_ok=True);folder=OUT/'jobs'/name;folder.mkdir(parents=True,exist_ok=False)
 d=read(REPORTS/'EXPERIMENT_REGISTRY.json');spent=sum(j.get('seconds',0)/3600 for j in d['jobs'].values() if j.get('GPU') is not None)
 active=sum(max(0,time.time()-j.get('start_epoch',time.time()))/3600 for j in d['jobs'].values() if j.get('GPU') is not None and j.get('status')=='RUNNING')
 assert spent+active<read(REPORTS/'PREREGISTRATION.json')['gpu_hours_max']
 assert shutil.disk_usage('/data1').free>=30*2**30
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TQDM_DISABLE='1')
 command=[PYTHON,str(ROOT/'reproduction_tools'/script),*args];begin=time.time()
 entry=dict(status='RUNNING',GPU=gpu,command=command,start_epoch=begin,source=binding(name),log=str(folder/'OUTPUT.log'))
 with (folder/'OUTPUT.log').open('w') as log:
  p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);entry['PID']=p.pid;update(name,entry)
  print('STARTED',name,p.pid,gpu,flush=True)
  try:code=p.wait(timeout=min(7200,(24-spent-active)*3600))
  except subprocess.TimeoutExpired:
   p.terminate();p.wait(timeout=30);code=-15;entry['budget_timeout']=True
 entry.update(status='COMPLETE' if code==0 else 'FAILED',exit_code=code,seconds=time.time()-begin,log=ref(folder/'OUTPUT.log'));update(name,entry)
 print('TERMINAL',name,entry['status'],entry['seconds'],flush=True)
 if code:raise SystemExit(code)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--script',required=True);a,args=p.parse_known_args();main(a.name,a.gpu,a.script,args)
