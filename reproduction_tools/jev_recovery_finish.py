"""Complete finite identity/paired/real-FPS evidence after frozen learning routes."""
import argparse
from jev_recovery_common import *
from jev_recovery_controller import run_job
from jev_recovery_publish import publish
def main(gpu):
 protect()
 while True:
  p=OUT/'progression/PROGRESS.json'
  if p.exists() and read(p).get('status')=='BOUNDED_TRAIN_ONLINE_ROUTES_COMPLETE':break
  if unreviewed_failures():raise RuntimeError('Retained predecessor failure needs review')
  save(OUT/'finish/PROGRESS.json',dict(status='WAITING_FROZEN_LEARNING_AND_ONLINE',GPU=gpu));time.sleep(30)
 decisions=read(p)['decisions'];eff=[];identity=[]
 for decision in decisions:
  revision=decision['revision'];base=decision['base'];phase='formal' if decision['formal_eligible'] else 'pilot'
  run_job(f'R{revision}_{base}_question_mechanism_{phase}','jev_recovery_question_probe.py',['--revision',revision,'--base',base,'--phase',phase],gpu)
  if phase=='formal':run_job(f'R{revision}_{base}_fixed_query_retrieval','jev_recovery_retrieval.py',['--revision',revision,'--base',base],None)
  run_job(f'R{revision}_{base}_paired_{phase}','jev_recovery_online.py',['--kind','paired','--arm','jev','--revision',revision,'--base',base,'--phase',phase,'--videos',16],gpu)
  for arm in ('parent','jev','threshold','mlp','attention'):
   # Every pilot has one complete native TRAIN16. Promoted methods also have
   # full formal three-seed DEV; identity metrics are from actual native IDs.
   for seed in (SEEDS if phase=='formal' else [20261009]):
    for video in (DEV if phase=='formal' else [16]):
     run_job(f'R{revision}_{base}_{arm}_identity_{phase}_{seed}_{video}','jev_recovery_online.py',['--kind','identity','--arm',arm,'--revision',revision,'--base',base,'--seed',seed,'--phase',phase,'--videos',video],None)
     file=REPORTS/f'R{revision}_{base}'/f'IDENTITY_{arm}_{phase}_SEED{seed}_VIDEO{video:02d}.json';identity.append(ref(file))
   # All timing on same physical GPU, idle except this process. Cached
   # training/online jobs have finished. Never steal another process's device.
   uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
   while True:
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    if not any(line.startswith(uuid+',') for line in processes.splitlines()):break
    save(OUT/'finish/PROGRESS.json',dict(status='WAITING_UNCONTENDED_GPU',GPU=gpu,revision=revision,arm=arm));time.sleep(30)
   run_job(f'R{revision}_{base}_{arm}_FPS_{phase}','jev_recovery_online.py',['--kind','efficiency','--arm',arm,'--revision',revision,'--base',base,'--phase',phase,'--videos',17],gpu)
   file=REPORTS/f'R{revision}_{base}'/f'EFFICIENCY_{arm}_{phase}_SEED20261009_VIDEO17.json';eff.append(ref(file));publish(f'Record R{revision} {base} {arm} three uncontended full-image FPS restarts')
   if phase=='formal':
    run_job(f'R{revision}_{base}_{arm}_parameters','jev_recovery_online.py',['--kind','parameters','--arm',arm,'--revision',revision,'--base',base,'--videos',12],gpu)
  if phase=='formal':
   for arm in ('jev','threshold','mlp','attention'):
    os.environ['JEV_RECOVERY_REVISION']=str(revision);os.environ['JEV_RECOVERY_BASE']=base
    run_job(f'R{revision}_{base}_{arm}_components','jev_recovery_component_profile.py',['--arm',arm],gpu)
   publish(f'Record R{revision} {base} complete component profiles and parameter reads')
 save(REPORTS/'PARAMETERS_AND_FPS.json',dict(status='COMPLETE_BOUNDED_PRIMARY_TIMINGS',efficiency=eff,GPU=gpu,partial_FLOPs_not_claimed_exact=True,failed_route_pilot_FPS_not_formal_FPS=True))
 save(REPORTS/'N01_N10_ANALYSIS.json',dict(status='COMPLETED_ACTUAL_NATIVE_CASE_REFERENCES',sources=identity,scope='Independent cold-run per-policy owner namespaces; paired same-state windows reported separately'))
 from jev_recovery_results import refresh
 refresh();save(OUT/'finish/READY.json',dict(status='COMPLETE_BOUNDED_EXECUTION_PENDING_SCIENTIFIC_REVIEW',routes=decisions,identity_cases=len(identity),FPS_methods=len(eff),MASTER_BREAKTHROUGH=False,legacy_goal_gate_unchanged=True))
 save(REPORTS/'NEXT_ACTION.json',dict(status='SCIENTIFIC_REVIEW_REQUIRED',action='Evaluate frozen three-route results against preregistered A/B/C goals, certify question/head/solver and historical XVI regressions; do not auto-declare success',ready=ref(OUT/'finish/READY.json')))
 publish('Record actual bounded recovery identity and efficiency evidence for scientific review')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=6);a=p.parse_args()
 try:main(a.gpu)
 except Exception:
  import traceback;failure('research_finish',traceback.format_exc());raise
