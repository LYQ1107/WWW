"""Single conditional cache check after all default timing jobs end."""
from gmt_jev_common import *
from gmt_jev_job import spent

def status(**fields):save(OUT/'learned_cache/QUEUE.json',dict(PID=os.getpid(),**fields))

def wait_defaults():
 begin=time.monotonic()
 while True:
  q=read(OUT/'learned_validation/QUEUE.json');p=pathlib.Path('/proc')/str(q['PID'])/'cmdline'
  try:signature=p.read_bytes()
  except FileNotFoundError:signature=b''
  live=b'gmt_jev_learned_validation_queue.py' in signature
  if not live:
   assert q['status']=='COMPLETE_FINITE_LEARNED_QUALITY_AND_FPS',q['status'];return
  assert time.monotonic()-begin<3*3600
  status(status='WAITING_VERIFIED_DEFAULT_QUEUE',observed_parent_PID=q['PID'])
  time.sleep(30)

def publish(message):
 subprocess.run([PYTHON,str(ROOT/'reproduction_tools/gmt_jev_research_report.py')],cwd=ROOT,check=True)
 subprocess.run([PYTHON,str(ROOT/'reproduction_tools/gmt_jev_publish.py'),'--message',message],cwd=ROOT,check=True)

def job(name,script,args):
 assert name not in read(REPORTS/'EXPERIMENT_REGISTRY.json')['jobs'],'Never duplicate an existing case'
 status(status='RUNNING_SINGLE_CACHE_ENGINEERING_CHECK',job=name)
 subprocess.run([PYTHON,str(ROOT/'reproduction_tools/gmt_jev_job.py'),'--name',name,'--gpu','7','--script',script,'--',*args],cwd=ROOT,check=True)

def main():
 try:
  wait_defaults()
  mlp=read(REPORTS/'SPEED_mlp_learned.json');original=read(REPORTS/'SPEED_original.json')
  assert len(mlp['trials'])==3 and mlp['GPU_UUID']==original['GPU_UUID']
  if mlp['mean_full_scene_FPS']>=.95*original['mean_full_scene_FPS']:
   status(status='SKIPPED_DEFAULT_MLP_ALREADY_MEETS_SPEED');return
  if spent()+.9>12:
   status(status='SKIPPED_INSUFFICIENT_REMAINING_GPU_BUDGET',GPU_hours=spent());return
  assert read(REPORTS/'LEARNED_GATE_RESULTS.json')['methods']['mlp']['joint_measured_quality_pass']
  job('P5_learned_MLP_cache_full_parity_GPU7','gmt_jev_cache_parity.py',['--kind','mlp_learned'])
  publish('Verify full native cached learned MLP equivalence without policy changes')
  if spent()+.7>12:
   status(status='SKIPPED_CACHE_TIMING_RESOURCE_PREFLIGHT',GPU_hours=spent());return
  job('P5_mlp_learned_cached_full_GPU7','gmt_jev_speed.py',['--kind','mlp_learned_cached','--gpu','7'])
  status(status='COMPLETE_SINGLE_CACHE_ENGINEERING_CHECK',scientific_success=False)
  publish('Record separate cached learned MLP complete image timing')
 except Exception as e:
  status(status='FAILED_RETAINED_NO_AUTOMATIC_RESTART',reason=str(e));raise

if __name__=='__main__':main()
