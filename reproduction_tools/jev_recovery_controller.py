"""Finite preregistered job controller; scientific gates never auto-complete MASTER."""
import argparse,fcntl,signal
from current_research_goal import redirect_if_superseded
if __name__=='__main__' and redirect_if_superseded('reports/JEV_RECOVERY/MASTER_GOAL.json'):
 raise SystemExit(0)
from jev_recovery_common import *

def register(name,update):
 with (OUT/'REGISTRY.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);r=read(REPORTS/'EXPERIMENT_REGISTRY.json')
  r['jobs'].setdefault(name,dict(status='PLANNED')).update(update);save(REPORTS/'EXPERIMENT_REGISTRY.json',r)

def run_job(name,script,args=(),gpu=None):
 locks=OUT/'job_locks';locks.mkdir(exist_ok=True)
 assert '/' not in name and name not in ('.','..')
 # Different finite queues may request the same immutable experiment. Hold
 # this lock through completion so only one launches it; later callers reuse it.
 with (locks/(name+'.lock')).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  return _run_job(name,script,args,gpu)

def _run_job(name,script,args=(),gpu=None):
 protect();r=read(REPORTS/'EXPERIMENT_REGISTRY.json');old=r['jobs'].get(name,{})
 if old.get('status')=='COMPLETE':return
 if old.get('status') in ('RUNNING','FAILED'):raise RuntimeError('Existing running/failed job needs explicit recorded recovery: '+name)
 env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
 log=OUT/'controller_logs'/f'{name}.log';log.parent.mkdir(exist_ok=True)
 command=[PYTHON,str(ROOT/'reproduction_tools'/script),*map(str,args)]
 save(REPORTS/'NEXT_ACTION.json',dict(status='EXECUTING',experiment_id=name,command=command,GPU=gpu,log=str(log),MASTER='ACTIVE_RESEARCH'))
 with log.open('x') as f:
  p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
  register(name,dict(status='RUNNING',PID=p.pid,GPU=gpu,command=command,log=str(log),source=binding(name)))
  while p.poll() is None:
   progress=dict(status='EXECUTING',job=name,PID=p.pid,log=str(log),alive=pathlib.Path('/proc',str(p.pid)).exists(),updated_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
   save(OUT/'controller'/name/'PROGRESS.json',progress)
   save(OUT/'controller/PROGRESS.json',progress)
   time.sleep(20)
 register(name,dict(status='COMPLETE' if p.returncode==0 else 'FAILED',exit_code=p.returncode,log=ref(log)))
 if p.returncode:raise RuntimeError('Retained failed job: '+name)

def r1(gpu):
 protect();dependencies=['QUESTION_DISAGREEMENT_CENSUS.json','QUESTION_EFFECTIVE_SUPERVISION.json','LEGAL_ASSOCIATION_COVERAGE.json']
 while not all((REPORTS/n).exists() for n in dependencies):
  if unreviewed_failures():raise RuntimeError('Audit dependency failed; review retained failure')
  save(OUT/'controller/PROGRESS.json',dict(status='WAITING_AUDIT',dependencies=dependencies));time.sleep(20)
 from jev_recovery_publish import publish
 publish('Record complete TRAIN disagreement census and legal candidate coverage')
 run_job('R1_contract','jev_recovery_tests.py',gpu=gpu)
 publish('Verify zero residual native parity on XIV and strong XIII Fixed bases')
 for arm in ('jev','threshold','mlp','attention'):
  phase='tiny8' if arm=='jev' else 'tiny'
  run_job('R1_'+arm+'_'+phase,'jev_recovery_train.py',['--arm',arm,'--phase',phase],gpu)
  tiny=read(REPORTS/f'TRAIN_R1_XIV_{arm}_{phase}_SEED20261009.json')
  assert tiny['engineering_training_pass'],'Tiny failed for '+arm
 publish('Record equal-supervision Tiny128 learning on expanded legal plans')
 for arm in ('jev','threshold','mlp','attention'):
  run_job('R1_'+arm+'_pilot','jev_recovery_train.py',['--arm',arm,'--phase','pilot'],gpu)
  publish('Record R1 '+arm+' Pilot1000 with certified conflict sampling')
 for arm in ('parent','jev','threshold','mlp','attention'):
  run_job('R1_'+arm+'_TRAIN16_online','jev_recovery_online.py',['--arm',arm,'--videos',16],gpu)
  publish('Record complete mutated-state TRAIN16 audit for R1 '+arm)
 from jev_recovery_gate import evaluate
 decision=evaluate(1,'XIV')
 publish('Freeze R1 TRAIN mechanism gate and next bounded research action')
 save(OUT/'controller/PROGRESS.json',dict(status='R1_PILOT_GATE_COMPLETE',decision=decision,next_action='Formal/R2/R3 executor follows frozen TRAIN gate; MASTER remains ACTIVE'))
 if decision['formal_eligible']:
  save(REPORTS/'NEXT_ACTION.json',dict(status='READY_FOR_FORMAL',revision=1,base='XIV',action='Equal 4000 cumulative updates, all three fixed seeds, then frozen complete DEV',decision=ref(REPORTS/'R1_XIV_MECHANISM_GATE.json')))
 else:save(REPORTS/'NEXT_ACTION.json',dict(status='READY_FOR_R2',action='Proceed typed evidence-preserving fusion, do not end research on R1 negative result',decision=ref(REPORTS/'R1_XIV_MECHANISM_GATE.json')))

def status():
 r=read(REPORTS/'EXPERIMENT_REGISTRY.json');return dict(goal=read(REPORTS/'MASTER_GOAL.json'),next_action=read(REPORTS/'NEXT_ACTION.json'),jobs=r['jobs'],progress={str(p.relative_to(OUT)):read(p) for p in OUT.rglob('PROGRESS.json')})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--status',action='store_true');p.add_argument('--r1',action='store_true');p.add_argument('--gpu',type=int,default=6);a=p.parse_args()
 if a.status:print(json.dumps(status(),indent=2))
 elif a.r1:
  try:r1(a.gpu)
  except Exception:
   import traceback;failure('controller',traceback.format_exc());raise
