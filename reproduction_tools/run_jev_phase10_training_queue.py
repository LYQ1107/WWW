"""Equal optimizer budgets for all architecture/supervision/seed combinations."""
import argparse,json,os,subprocess,time
from jev_phase10_common import *

def main():
 protect();assert not binding()['dirty'];assert json.loads((REPORTS/'DATA_ELIGIBILITY.json').read_text())['gate_pass'];assert json.loads((REPORTS/'TINY_OVERFIT.json').read_text())['status']=='PASS'
 names=['CandidateMLP','CandidateDeepSets','CandidateJEV','CandidateJEV_without_QA','CandidateJEV_without_context'];sup=['correctness','H32_ranking','joint'];seeds=[20261008,20261009,20261010];slots=[4,5,6,7,1,9];logroot=OUT/'execution_logs';completed=[]
 for budget in [20,50,100]:
  alltasks=[(name,s,seed)for name in names for s in sup for seed in seeds];queues={g:[]for g in slots}
  for index,task in enumerate(alltasks):queues[slots[index%len(slots)]].append(task)
  running={};failed=[]
  while any(queues.values())or running:
   for gpu in slots:
    if gpu in running or not queues[gpu]or failed:continue
    name,s,seed=queues[gpu].pop(0);log=logroot/f'train_{name}_{s}_{seed}_{budget:03d}.log';handle=log.open('a');env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');p=subprocess.Popen([PYTHON,'reproduction_tools/train_jev_phase10_candidates.py','--model',name,'--supervision',s,'--seed',str(seed),'--epochs',str(budget)],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT);running[gpu]={'p':p,'handle':handle,'task':{'model':name,'supervision':s,'seed':seed,'epochs':budget,'GPU':gpu,'log':str(log)}};print('TRAIN_START',running[gpu]['task'],flush=True)
   for gpu,r in list(running.items()):
    code=r['p'].poll()
    if code is None:continue
    r['handle'].close();item={**r['task'],'exit_code':code};(completed if code==0 else failed).append(item);del running[gpu];print('TRAIN_FINISH',item,flush=True)
   save(OUT/'TRAINING_EXECUTION.json',{'status':'FAIL'if failed else'RUNNING','budget':budget,'pending':{str(g):q for g,q in queues.items()},'running':[{**r['task'],'PID':r['p'].pid}for r in running.values()],'completed':completed,'failed':failed,'source_commit':binding()['source_commit']})
   if failed and not running:break
   if any(queues.values())or running:time.sleep(5)
  assert not failed,failed
  under=[]
  for name in names[:3]:
   for seed in seeds:
    p=OUT/'fair_training_v1'/name/'correctness'/f'seed{seed}'/f'RESULT_{budget:03d}.json';r=json.loads(p.read_text());fit=r['history'][-1]['TRAIN']['group_weighted_rank1']
    if fit is None or fit<.95:under.append({'model':name,'seed':seed,'last_epoch_TRAIN_rank1':fit})
  decision={'budget':budget,'all45_runs_complete':True,'underfit_primary_correctness':under,'extend_every_run_equally':bool(under)and budget<100,'no_validation_or_heldout_extension_selection':True};save(OUT/f'TRAINING_BUDGET_{budget:03d}.json',decision)
  if not under or budget==100:break
 save(OUT/'TRAINING_COMPLETE.json',{'status':'COMPLETE','models':names,'supervisions':sup,'seeds':seeds,'final_common_epochs':budget,'all45_runs_complete':True,'extension_decision':decision,'source_commit':binding()['source_commit'],'heldout_sealed':True});print('ALL_FAIR_TRAINING_COMPLETE',budget,flush=True)
if __name__=='__main__':main()
