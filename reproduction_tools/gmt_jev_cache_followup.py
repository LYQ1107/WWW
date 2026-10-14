"""Finite optional-cache parity and timing after the original queue finishes."""
from gmt_jev_common import *
from gmt_jev_speed_queue import job,publish_step

def main():
 begin=time.monotonic()
 while True:
  q=read(OUT/'speed/QUEUE.json')
  if q['status']=='COMPLETE_FINITE_SPEED_QUEUE':break
  p=pathlib.Path('/proc')/str(q['PID'])/'cmdline'
  assert p.exists() and b'gmt_jev_speed_queue.py' in p.read_bytes(),'Original queue missing: audit terminal jobs before continuing'
  assert time.monotonic()-begin<8*3600
  save(OUT/'speed/CACHE_FOLLOWUP.json',dict(status='WAITING_VERIFIED_ORIGINAL_QUEUE',PID=os.getpid(),observed_queue_PID=q['PID']))
  time.sleep(30)
 # Exclusive same-GPU trials stay separate from any new parity workload.
 save(OUT/'speed/CACHE_FOLLOWUP.json',dict(status='RUNNING_FULL_NATIVE_CACHE_PARITY',PID=os.getpid()))
 job('P5_feature_cache_full_native_parity_GPU7','gmt_jev_cache_parity.py',7)
 publish_step('exact optional feature cache full native parity')
 save(OUT/'speed/CACHE_FOLLOWUP.json',dict(status='RUNNING_CACHED_RULE_FULL_IMAGE_TIMING',PID=os.getpid()))
 job('P5_rule_cached_full_image_GPU7','gmt_jev_speed.py',7,['--kind','rule_cached','--gpu','7'])
 publish_step('optional cache Rule full-image timing')
 save(OUT/'speed/CACHE_FOLLOWUP.json',dict(status='COMPLETE_FINITE_CACHE_FOLLOWUP',PID=os.getpid()))

if __name__=='__main__':main()
