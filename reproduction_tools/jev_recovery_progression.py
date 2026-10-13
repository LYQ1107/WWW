"""Continue bounded scientific routes after R1; never stop merely at first NO_GO.

TRAIN determines promotion. All revision designs freeze before new DEV results.
DEV is development evaluation only and cannot tune/select checkpoints or seeds.
"""
import argparse
from jev_recovery_common import *
from jev_recovery_controller import run_job
from jev_recovery_publish import publish
from jev_recovery_gate import evaluate

def stage(revision,base,arms,gpu):
 contract=REPORTS/f'R{revision}_MODEL_AND_NATIVE_CONTRACT_TESTS.json'
 if not contract.exists():run_job(f'R{revision}_contract','jev_recovery_tests.py',['--revision',revision],gpu)
 for phase in ('tiny','pilot'):
  for arm in arms:
   run_job(f'R{revision}_{base}_{arm}_{phase}','jev_recovery_train.py',['--arm',arm,'--phase',phase,'--revision',revision,'--base',base],gpu)
   t=read(REPORTS/f'TRAIN_R{revision}_{base}_{arm}_{phase}_SEED20261009.json')
   assert t['engineering_training_pass'],f'{arm} {phase} engineering failed; retain and review'
   publish(f'Record R{revision} {base} {arm} {phase} equal-budget training')
 for arm in ('parent',*arms):
  run_job(f'R{revision}_{base}_{arm}_TRAIN16','jev_recovery_online.py',['--arm',arm,'--revision',revision,'--base',base,'--videos',16],gpu)
  publish(f'Record complete R{revision} {base} {arm} TRAIN16 native controller audit')
 decision=evaluate(revision,base);publish(f'Freeze R{revision} {base} TRAIN-only mechanism decision');return decision

def formal(revision,base,arms,gpu):
 for seed in SEEDS:
  for arm in arms:
   if seed!=20261009:
    run_job(f'R{revision}_{base}_{arm}_pilot_seed{seed}','jev_recovery_train.py',['--arm',arm,'--phase','pilot','--seed',seed,'--revision',revision,'--base',base],gpu)
   run_job(f'R{revision}_{base}_{arm}_formal_seed{seed}','jev_recovery_train.py',['--arm',arm,'--phase','formal','--seed',seed,'--revision',revision,'--base',base],gpu)
   publish(f'Record R{revision} {base} {arm} seed{seed} frozen last 4000-update weights')
 # All weights finish before any DEV outcome is observed.
 save(REPORTS/f'R{revision}_{base}_DEV_FREEZE.json',dict(status='FROZEN_BEFORE_NEW_DEV',revision=revision,base=base,arms=list(arms),seeds=SEEDS,weights=[ref(OUT/f'training_R{revision}'/base/a/f'seed{s}/formal/RESULT.json') for s in SEEDS for a in arms],checkpoint_selection='last',DEV_tuning=False))
 publish(f'Freeze all R{revision} {base} three-seed models before complete DEV evaluation')
 for seed in SEEDS:
  for arm in ('parent',*arms):
   for video in DEV:
    run_job(f'R{revision}_{base}_{arm}_seed{seed}_DEV{video}','jev_recovery_online.py',['--arm',arm,'--seed',seed,'--phase','formal','--revision',revision,'--base',base,'--videos',video],gpu)
   run_job(f'R{revision}_{base}_{arm}_seed{seed}_official','jev_recovery_online.py',['--kind','matlab','--arm',arm,'--seed',seed,'--phase','formal','--revision',revision,'--base',base,'--videos',*DEV],gpu)
   from jev_recovery_results import refresh
   refresh();publish(f'Record pooled R{revision} {base} {arm} seed{seed} official full online metrics')

def main(gpu):
 protect()
 while not (REPORTS/'R1_XIV_MECHANISM_GATE.json').exists():
  if unreviewed_failures():raise RuntimeError('Predecessor failure requires retained-log review')
  save(OUT/'progression/PROGRESS.json',dict(status='WAITING_R1_TRAIN_GATE',GPU=gpu));time.sleep(20)
 # Wait for original R1 controller to release GPU and finish its publication.
 while True:
  registry=read(REPORTS/'EXPERIMENT_REGISTRY.json')
  busy=[name for name,j in registry['jobs'].items() if j['status']=='RUNNING' and j.get('GPU')==gpu]
  if not busy:break
  save(OUT/'progression/PROGRESS.json',dict(status='WAITING_OWN_GPU_PREDECESSOR',GPU=gpu,jobs=busy));time.sleep(20)
 decisions=[read(REPORTS/'R1_XIV_MECHANISM_GATE.json')]
 while not all((REPORTS/n).exists() for n in ['R2_QUESTION_SUPERVISION.json','R2_ANCHOR_SUPERVISION.json','R2_EARLY_EVIDENCE.json','R2_CANDIDATE_AND_QUESTION_AUDIT.json','R2_EARLY_EVIDENCE_CONTRACT.json']):
  if unreviewed_failures():raise RuntimeError('R2 label audit failed')
  time.sleep(20)
 # Revision 2 was predeclared before any new DEV: broad core/ordinary/fusion and
 # required question/shared-compute ablations, with fixed uniform tiny/pilot.
 r2arms=['jev','threshold','mlp','attention','f0_mean','mlp_fusion','attention_fusion','fixed_question','cross_only','stability_only','no_interaction','no_shared']
 decisions.append(stage(2,'XIV',r2arms,gpu))
 # Decide what to migrate from TRAIN evidence, never from new DEV performance.
 r2=decisions[-1];good=r2['rows']['jev']['current_mechanism'] and r2['rows']['jev']['no_severe_online_collapse']
 source_revision=2 if good else 1
 preregistration=dict(status='FROZEN_BEFORE_STRONG_BASE_NEW_GRADIENTS',base='XIII',mechanism_revision=source_revision,source_gate=ref(REPORTS/'R2_XIV_MECHANISM_GATE.json'),reason='Transfer typed mechanism only if TRAIN JEV mechanism qualifies; otherwise transfer bounded proposal/supervision correction alone',arms=['jev','threshold','mlp','attention'],seeds=SEEDS,DEV_selection=False)
 existing=REPORTS/'R3_PREREGISTRATION.json'
 if existing.exists():assert read(existing)==preregistration,'Preserve frozen R3 protocol; resume cannot change it'
 else:save(existing,preregistration)
 decisions.append(stage(3,'XIII',['jev','threshold','mlp','attention'],gpu))
 # A negative route gets a full report and the next scientifically grounded
 # route, rather than more steps on that failed hypothesis.
 for decision in decisions:
  if decision['formal_eligible']:
   arms=['jev','threshold','mlp','attention']
   if decision['revision']==2:arms=r2arms
   formal(decision['revision'],decision['base'],arms,gpu)
 from jev_recovery_results import refresh
 refresh();save(OUT/'progression/PROGRESS.json',dict(status='BOUNDED_TRAIN_ONLINE_ROUTES_COMPLETE',decisions=decisions,MASTER='ACTIVE_UNTIL_IDENTITY_EFFICIENCY_AND_SCIENTIFIC_REVIEW',next_action='Matched same-state mechanism, identity regression and uncontended real-image FPS; no TEST'))
 save(REPORTS/'NEXT_ACTION.json',dict(status='EVIDENCE_REVIEW_REQUIRED',action='Review full three-route evidence and finish frozen same-state/identity/FPS gates; MASTER not automatically complete',decisions=[ref(REPORTS/f"R{d['revision']}_{d['base']}_MECHANISM_GATE.json") for d in decisions]))
 publish('Record bounded multi-route recovery evidence without declaring MASTER success')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=6);a=p.parse_args()
 try:main(a.gpu)
 except Exception:
  import traceback;failure('research_progression',traceback.format_exc());raise
