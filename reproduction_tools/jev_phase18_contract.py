"""Real native same-state restore and capacity checks on one frozen TRAIN prefix."""
from jev_phase18_common import *
import copy,gzip
import torch
from jev_phase16_common import load_policy
from jev_phase13_runtime import build_tracker,cache_inputs,run
from jev_phase15_pilot_native import converted_prefix
from jev_phase14_artifacts import load_dense
from gtr.modeling.jev_phase15.native_commit_adapter import attach
from gtr.modeling.jev_phase18.constraint import NativeConstraint,constrained_choice
from gtr.modeling.jev_stage2.assignment import lawful_choice
from gtr.modeling.jev_native_state import fingerprint
if __name__=='__main__':
 protect();torch.set_num_threads(1);torch.manual_seed(20261009);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 x=torch.tensor([[5.,4.,0.],[9.,3.,0.],[1.,8.,0.]],device='cuda');m=torch.ones(3,2,dtype=torch.bool,device='cuda');before=x.clone();original=lawful_choice(x,m);locked=constrained_choice(x,m,0,0);assert torch.equal(x,before) and locked[0]==0 and locked.count(0)==1
 old=read(Path('/home/liuyeqiang/WWW_jev_phase17_runtime/20261011_v1/factorial/prefix/B0/video12/frame000003_view1_row6/RESULT.json'));entry=old['prefix'];prefix=converted_prefix(load_dense(entry['prefix']['path'],map_location='cuda:0'));policy,checkpoint,_=load_policy('v3');values,frames,reader=cache_inputs(12);results=[]
 for variant in ['baseline','forced']:
  model=build_tracker(12,policy,react_learned=False);executor=attach(model);trace=[];queries=[]
  executor.observer=lambda **d:queries.append(dict(key=[d['context']['frame'],d['context']['view']],task=d['task'],refs=d['refs'],scores=d['logits'].cpu().tolist(),legal=d['batch']['legal'][0].cpu().tolist()))
  executor.commit_observer=lambda **d:trace.append(dict(key=[d['frame'],d['view']],ids=d['instances'][-1].track_ids.cpu().tolist()))
  if variant=='baseline':
   with torch.no_grad():run(model,values,frames,stop=18,prefix=copy.deepcopy(prefix))
   base=trace;first=queries[0];row=6;selected=lawful_choice(torch.tensor(first['scores']),torch.tensor(first['legal']))[row];reference=next(ref for col,ref in enumerate(first['refs']) if col!=selected and first['legal'][row][col])
   with gzip.open(old['commits']['path'],'rt') as f:expected=[json.loads(line) for line in f if json.loads(line)['key'][0]<19]
   assert trace==[dict(key=e['key'],ids=e['ids']) for e in expected],'real prefix baseline does not match frozen XVII'
  else:
   with NativeConstraint(executor,(3,1),row,reference) as constraint,torch.no_grad():run(model,values,frames,stop=18,prefix=copy.deepcopy(prefix))
   assert constraint.fired==1 and trace[0]['ids'][row]==reference and queries[0]==first,'native constraint changed first raw score/mask/ref or failed to commit'
  results.append(dict(variant=variant,starting_native_fp=fingerprint(prefix),first_score=queries[0],commit_trace=trace,constraint_audit=constraint.audit if variant=='forced' else None))
  del model,executor;torch.cuda.empty_cache()
 assert results[0]['starting_native_fp']==results[1]['starting_native_fp']
 save(REPORTS/'NATIVE_CONSTRAINT_CONTRACT.json',dict(status='PASS_REAL_NATIVE_PARITY',binding=binding(checkpoint=checkpoint),synthetic_capacity_check=dict(original=original,constrained=locked,scores_unchanged=True),actual_cases=results,source=ref(Path(old['raw_predictions']['path']).parent/'RESULT.json'),GT_actor_inputs=False,new_Match_second_scoring_validation=False))
 print('NATIVE_CONSTRAINT_CONTRACT_PASS',flush=True)
