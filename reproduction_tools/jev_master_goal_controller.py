"""Recoverable, evidence-hashed research progression; no implicit training/test unlock."""
import argparse,subprocess
from pathlib import Path
from current_research_goal import redirect_if_superseded
if redirect_if_superseded('reports/JEV_MASTER/MASTER_GOAL.json'):
 raise SystemExit(0)
if (Path(__file__).resolve().parents[1]/'reports/JEV_MASTER/MASTER_COMPLETION_REQUIREMENTS_AUDIT.json').exists():
 from jev_master_audit_common import *
elif (Path(__file__).resolve().parents[1]/'reports/JEV_PHASE21/PREREGISTRATION.json').exists():
 from jev_phase21_common import *
elif (Path(__file__).resolve().parents[1]/'reports/JEV_PHASE20/PREREGISTRATION.json').exists():
 from jev_phase20_common import *
elif (Path(__file__).resolve().parents[1]/'reports/JEV_PHASE19/PREREGISTRATION.json').exists():
 from jev_phase19_common import *
elif (Path(__file__).resolve().parents[1]/'reports/JEV_PHASE18/PREREGISTRATION.json').exists():
 from jev_phase18_common import *
else:
 from jev_phase17_common import *
TERMINAL={'COMPLETE','NO_GO','NOT_RUN','BLOCKED_EVIDENCE'}
def validate_evidence(paths):
 for item in paths:
  assert sha(item['path'])==item['SHA256'],('evidence changed',item['path'])
def tick(execute=False):
 protect();goal=read(MASTER/'MASTER_GOAL.json');registry=read(MASTER/'EXPERIMENT_REGISTRY.json')
 validate_evidence(read(MASTER/'EVIDENCE_REGISTRY.json')['entries'])
 by_id={e['id']:e for e in registry['experiments']};active=[]
 for e in registry['experiments']:
  validate_evidence(e['evidence'])
  if e['status'] in TERMINAL:continue
  if e['status'] in {'RUNNING','EXECUTED_AWAITING_EVIDENCE_REVIEW'}:
   progress=Path(e.get('progress_path','/missing'));result=Path(e.get('result_path','/missing'))
   active.append(dict(experiment=e['id'],progress=read(progress) if progress.exists() else None,result_present=result.exists(),status=e['status']))
   # An external live task is not a missing implementation. Continue independent work.
   continue
  if not all(by_id[i]['status'] in TERMINAL for i in e['requires']):continue
  if e['requires_learning_gate']:
   gate=REPORTS/'LEARNING_ELIGIBILITY.json'
   if not gate.exists() or read(gate).get('qualified') is not True:
    e.update(status='NOT_RUN',reason='No predeclared learning eligibility; no training executed',metrics=None)
    continue
  action=dict(status='READY' if e['commands'] else 'NEEDS_EXECUTABLE_IMPLEMENTATION',experiment=e['id'],reason=e['description'],commands=e['commands'])
  save(MASTER/'NEXT_ACTION.json',action);goal.update(current_phase_status=e['id']+'_'+action['status'],next_experiment=e['id']);save(MASTER/'MASTER_GOAL.json',goal)
  save(MASTER/'EXPERIMENT_REGISTRY.json',registry)
  if execute and e['commands']:
   e['status']='RUNNING';save(MASTER/'EXPERIMENT_REGISTRY.json',registry)
   log=OUT/'controller'/e['id'];log.mkdir(parents=True,exist_ok=True)
   for n,cmd in enumerate(e['commands']):
    bounded_audits={'PUBLIC_RELEASE_METADATA_AUDIT':'jev_master_public_release_audit.py','REUSED_WILDTRACK_TRAIN_LABEL_AUDIT':'jev_master_wildtrack_labels.py','WILDTRACK_OFFLINE_GT_INTERFACE':'jev_master_wildtrack_gt_reader.py','SCOUT_ACCESS_TERMS_AUDIT':'jev_master_scout_access.py'}
    source_audit = e['id'] in bounded_audits and cmd==[PYTHON,str(ROOT/'reproduction_tools'/bounded_audits[e['id']])]
    assert source_audit or (cmd[0]==PYTHON and Path(cmd[1]).name.startswith('jev_phase'+str(goal.get('current_phase_number',17))+'_')),cmd
    with (log/f'attempt_{time.time_ns()}_{n}.log').open('w') as stream:
     p=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    if p.returncode:
     e.update(status='ENGINEERING_FAIL',returncode=p.returncode);save(MASTER/'EXPERIMENT_REGISTRY.json',registry);return action
   e['status']='EXECUTED_AWAITING_EVIDENCE_REVIEW';save(MASTER/'EXPERIMENT_REGISTRY.json',registry)
  return action
 save(MASTER/'EXPERIMENT_REGISTRY.json',registry)
 if not active and goal['master_status']=='PIVOT_REQUIRED':
  return read(MASTER/'NEXT_ACTION.json')
 action=dict(status='WAIT_FOR_EVIDENCE_REVIEW' if active else 'ROUND_COMPLETE_REQUIRES_SCIENTIFIC_REVIEW',active=active,master_status=goal['master_status'],scientific_success=False)
 save(MASTER/'NEXT_ACTION.json',action);return action
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--execute-next',action='store_true');a=p.parse_args();print(json.dumps(tick(a.execute_next),indent=2))
