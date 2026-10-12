"""Early full TRAIN16 controls, same immutable IDs later reused by progression."""
import argparse
from jev_recovery_common import *
from jev_recovery_controller import run_job
from jev_recovery_publish import publish
def main(gpu):
 protect()
 for arm in ['parent','jev','threshold','mlp','attention']:
  if arm!='parent':
   path=OUT/'training_R2/XIV'/arm/'seed20261009/pilot/RESULT.json';assert read(path)['status']=='COMPLETE'
  run_job('R2_XIV_'+arm+'_TRAIN16','jev_recovery_online.py',['--revision',2,'--base','XIV','--arm',arm,'--videos',16],gpu)
  publish('Record early complete R2 '+arm+' TRAIN16 reused by frozen mechanism gate')
 save(OUT/'native_main_queue/PROGRESS.json',dict(status='COMPLETE',GPU=gpu,no_duplicate_experiments=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=5);a=p.parse_args()
 try:main(a.gpu)
 except Exception:
  import traceback;failure('parallel_native_main_queue',traceback.format_exc());raise
