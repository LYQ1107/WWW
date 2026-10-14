"""Finite full TRAIN diagnostics after all same-GPU performance work."""
from gmt_jev_common import *
from gmt_jev_speed_queue import job,publish_step

def main():
 begin=time.monotonic()
 while True:
  q=read(OUT/'speed/CACHE_FOLLOWUP.json')
  if q['status']=='COMPLETE_FINITE_CACHE_FOLLOWUP':break
  p=pathlib.Path('/proc')/str(q['PID'])/'cmdline'
  assert p.exists() and b'gmt_jev_cache_followup.py' in p.read_bytes(),'Cache driver missing; audit terminal jobs before proceeding'
  assert time.monotonic()-begin<8*3600
  save(OUT/'opportunity_full/FOLLOWUP.json',dict(status='WAITING_VERIFIED_CACHE_DRIVER',PID=os.getpid(),observed_driver_PID=q['PID']))
  time.sleep(30)
 for video in [12,13,14]:
  save(OUT/'opportunity_full/FOLLOWUP.json',dict(status='RUNNING_FULL_TRAIN_AUDIT',PID=os.getpid(),video=video))
  job(f'P1_full_TRAIN{video}_coverage_GPU7','gmt_jev_full_train_audit.py',7,['--video',str(video)])
  publish_step(f'full TRAIN{video} fixed temporal opportunity diagnostic')
 reports=[read(REPORTS/f'FULL_TRAIN_OPPORTUNITY_VIDEO{v:02d}.json') for v in [12,13,14]]
 import collections
 counts=collections.Counter()
 for r in reports:counts.update(r['counts'])
 save(REPORTS/'FULL_TRAIN_OPPORTUNITY_AUDIT.json',dict(status='COMPLETE_ALL_REGISTERED_TRAIN_TEMPORAL_COVERAGE',cases=reports,counts=dict(counts),
  initial_prefix_preserved=True,no_training_started=True,independent_validation=False,source=binding()))
 publish_step('all registered TRAIN full temporal coverage summary')
 save(OUT/'opportunity_full/FOLLOWUP.json',dict(status='COMPLETE_FINITE_FULL_TRAIN_AUDITS',PID=os.getpid()))

if __name__=='__main__':main()
