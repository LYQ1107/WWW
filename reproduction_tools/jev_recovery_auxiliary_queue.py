"""Run predeclared causal diagnostics on an idle GPU without blocking native training."""
import argparse
from jev_recovery_common import *
from jev_recovery_controller import run_job
from jev_recovery_publish import publish
def main(gpu):
 protect()
 for revision,base in [(2,'XIV')]:
  required=[OUT/f'training_R{revision}'/base/a/'seed20261009/pilot/RESULT.json' for a in ['jev','threshold','mlp','attention']]
  while not all(p.exists() and read(p)['status']=='COMPLETE' for p in required):
   if unreviewed_failures():raise RuntimeError('Pilot dependency failed; retain original failure and review')
   save(OUT/'auxiliary/PROGRESS.json',dict(status='WAITING_FOUR_COMPLETED_PILOTS',required=[str(p) for p in required],GPU=gpu));time.sleep(30)
  run_job('R2_XIV_same_solver_budget_pilot','jev_recovery_solver_audit.py',['--revision',2,'--base',base],gpu)
  publish('Record exact same-state scorer solver and common-budget attribution on TRAIN16')
 save(OUT/'auxiliary/PROGRESS.json',dict(status='COMPLETE',GPU=gpu))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=2);a=p.parse_args()
 try:main(a.gpu)
 except Exception:
  import traceback;failure('auxiliary_attribution_queue',traceback.format_exc());raise
