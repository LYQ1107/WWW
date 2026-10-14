"""Finite same-GPU timing continuation; never launch duplicate or old research jobs."""
from gmt_jev_common import *
from gmt_jev_job import spent

def progress(**fields):
 save(OUT/'speed/QUEUE.json',dict(PID=os.getpid(),physical_GPU=7,**fields))

def await_original():
 name='P5_original_full_image_GPU7_retry';begin=time.monotonic()
 while True:
  job=read(REPORTS/'EXPERIMENT_REGISTRY.json')['jobs'][name]
  if job['status']!='RUNNING':
   assert job['status']=='COMPLETE',job['status'];return
  path=pathlib.Path('/proc')/str(job['PID'])/'cmdline'
  signature=path.read_bytes().replace(b'\0',b' ') if path.exists() else b''
  live=b'gmt_jev_speed.py --kind original --gpu 7' in signature
  if not live:
   # Give the registered parent time to collect an actual terminal child result.
   time.sleep(1)
   terminal=read(REPORTS/'EXPERIMENT_REGISTRY.json')['jobs'][name]
   assert terminal['status']!='RUNNING','Missing original child; audit parent before retry'
   assert terminal['status']=='COMPLETE',terminal['status'];return
  assert time.monotonic()-begin<3*3600
  reserve();assert spent()<12
  progress(status='WAITING_VERIFIED_ORIGINAL_CHILD',observed_child_PID=job['PID'],wait_seconds=time.monotonic()-begin)
  time.sleep(30)

def job(name,script,gpu=None,args=()):
 registry=read(REPORTS/'EXPERIMENT_REGISTRY.json')['jobs']
 if name in registry:
  assert registry[name]['status']=='COMPLETE','Existing job is not complete; never duplicate'
  return
 command=[PYTHON,str(ROOT/'reproduction_tools/gmt_jev_job.py'),'--name',name,'--script',script]
 if gpu is not None:command+=['--gpu',str(gpu)]
 command+=['--',*args]
 progress(status='RUNNING_FINITE_STEP',job=name)
 subprocess.run(command,cwd=ROOT,check=True)

def main():
 await_original()
 publish_step('original full-image timing')
 job('P5_original_stock_metrics','gmt_jev_stock_baseline.py')
 publish_step('matched original stock metrics')
 for kind in ['threshold','rule','confidence','mlp_zero','attention_zero','cic_zero']:
  job('P5_'+kind+'_full_image_GPU7','gmt_jev_speed.py',7,['--kind',kind,'--gpu','7'])
  publish_step(kind+' full-image timing')
 progress(status='COMPLETE_FINITE_SPEED_QUEUE',next='CONSOLIDATE_ACTUAL_METRICS_AND_PUBLISH')

def publish_step(phase):
 subprocess.run([PYTHON,str(ROOT/'reproduction_tools/gmt_jev_research_report.py')],cwd=ROOT,check=True)
 subprocess.run([PYTHON,str(ROOT/'reproduction_tools/gmt_jev_publish.py'),'--message','Record '+phase+' and refresh current GMT-JEV evidence'],cwd=ROOT,check=True)

if __name__=='__main__':main()
