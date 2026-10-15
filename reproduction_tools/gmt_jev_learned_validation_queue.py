"""Finite continuation: verify live smoke, all heads cold quality, actual FPS."""
from gmt_jev_common import *
from gmt_jev_job import spent

def state(**fields):
 save(OUT/'learned_validation/QUEUE.json',dict(PID=os.getpid(),physical_GPU=7,**fields))

def wait_smoke():
 name='P3_learned_gate_precheck_and_fixed300_GPU7';begin=time.monotonic()
 while True:
  job=read(REPORTS/'EXPERIMENT_REGISTRY.json')['jobs'][name]
  if job['status']!='RUNNING':
   assert job['status']=='COMPLETE', 'Real precheck/smoke failed; no online launch'
   assert read(REPORTS/'LEARNED_GATE_SMOKE_TRAINING.json')['status']=='COMPLETE'
   return
  path=pathlib.Path('/proc')/str(job['PID'])/'cmdline'
  signature=path.read_bytes() if path.exists() else b''
  if b'gmt_jev_learned_gate_smoke.py' not in signature:
   time.sleep(1);terminal=read(REPORTS/'EXPERIMENT_REGISTRY.json')['jobs'][name]
   assert terminal['status']!='RUNNING','Missing child; audit parent, never duplicate'
   assert terminal['status']=='COMPLETE';return
  reserve();assert spent()<12 and time.monotonic()-begin<3*3600
  state(status='WAITING_VERIFIED_REAL_SUPERVISION_SMOKE',observed_child_PID=job['PID'])
  time.sleep(30)

def publish(message):
 subprocess.run([PYTHON,str(ROOT/'reproduction_tools/gmt_jev_research_report.py')],cwd=ROOT,check=True)
 subprocess.run([PYTHON,str(ROOT/'reproduction_tools/gmt_jev_publish.py'),'--message',message],cwd=ROOT,check=True)

def run_job(kind,timing=False):
 prefix='P5' if timing else 'P4';name=f'{prefix}_{kind}_learned_full_GPU7'
 jobs=read(REPORTS/'EXPERIMENT_REGISTRY.json')['jobs']
 if name in jobs:
  assert jobs[name]['status']=='COMPLETE','Never duplicate a live or failed case';return
 # Explicit conservative preflight; actual registry monitor retains 12h hard limit.
 estimate=.8 if timing else .4
 if spent()+estimate>12:
  state(status='FINITE_QUEUE_RESOURCE_LIMIT',next_job=name,GPU_hours=spent(),estimated_next_GPU_hours=estimate)
  return False
 state(status='RUNNING_FIXED_ENDPOINT_VALIDATION',job=name)
 args=['--kind',kind+'_learned']+(['--gpu','7'] if timing else [])
 subprocess.run([PYTHON,str(ROOT/'reproduction_tools/gmt_jev_job.py'),'--name',name,'--gpu','7','--script',
  'gmt_jev_speed.py' if timing else 'gmt_jev_online.py','--',*args],cwd=ROOT,check=True)
 publish('Record fixed step300 '+kind+(' complete image FPS' if timing else ' native closed-loop quality'))
 return True

def main():
 try:
  wait_smoke();publish('Record certified TRAIN supervision and three fixed300 learned gates')
  for kind in ['mlp','attention','cic']:
   if run_job(kind) is False:return
  for kind in ['mlp','attention','cic']:
   if run_job(kind,timing=True) is False:return
  state(status='COMPLETE_FINITE_LEARNED_QUALITY_AND_FPS',scientific_success=False,
   next='Audit joint outcomes and independent validation qualification; no new data/architecture search')
  publish('Complete fair bounded learned gate quality and efficiency evidence')
 except Exception as e:
  state(status='FAILED_RETAINED_NO_AUTOMATIC_RESTART',reason=str(e),scientific_success=False)
  raise

if __name__=='__main__':main()
