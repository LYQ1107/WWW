"""Real TRAIN fixtures, constrained proposal reachability, and both native base parity."""
import gc,torch,traceback,argparse
from jev_recovery_common import *
from jev_recovery_learning import initialize,batch
from jev_oneweek_tests import fixture
from gtr.modeling.jev_recovery.plans import plans
from gtr.modeling.jev_stage2.assignment import lawful_choice

def main(revision=1,tag=None):
 protect();torch.set_num_threads(1)
 records=fixture();x,y=batch(records);results=[]
 # A real capacity conflict is re-solved, rather than independently overriding
 # rows. A deferred row with a legal old identity is represented by P1.
 z=torch.tensor([[2.,1.,0.],[1.9,1.,2.1]],device='cuda:0');mask=torch.ones(2,2,dtype=torch.bool,device='cuda:0')
 pp=plans(z,mask);assert pp[0]==(0,-1) and any(p[1]>=0 for p in pp)
 empty=plans(torch.zeros(2,1,device='cuda:0'),torch.zeros(2,0,device='cuda:0',dtype=torch.bool));assert empty==[(-1,-1)]
 for base in ('XIV','XIII'):
  arms=('threshold','mlp','attention','jev') if revision==1 else ('threshold','mlp','attention','jev','f0_mean','mlp_fusion','attention_fusion','fixed_question','cross_only','stability_only','no_interaction')
  for arm in arms:
   model,cp=initialize(arm,20261009,revision=revision,base=base);model.eval()
   with torch.no_grad():d=model.details(x);old=model.identity({k:v for k,v in x.items() if k!='commitment_features'})
   assert torch.equal(d['base_logits'],old) and torch.equal(d['edge_logits'],old)
   for i,r in enumerate(records):
    q=len(r['rows']);k=len(r['refs']);score=torch.cat([old[i,:q,:k],old[i,:q,-1:]],-1)
    assert d['plans'][i][int(d['plan_scores'][i].argmax())]==tuple(lawful_choice(score,x['legal'][i,:q,:k]))
   results.append(dict(base=base,arm=arm,maximum_plans=model.maximum,initial_logits_bitwise_exact=True,initial_choices_exact=True,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),initializer=cp))
   del model,d,old;gc.collect();torch.cuda.empty_cache()
 from jev_phase13_runtime import build_tracker,cache_inputs,run
 from jev_oneweek_online import FrozenPlanControl
 from gtr.modeling.jev_oneweek.native import attach_jevtrack
 from gtr.modeling.jev_native_state import fingerprint
 from audit_jev_stage2_gta_free import phase13_prefix
 from jev_oneweek_common import seed_all
 from jev_phase14_latency import prune
 values,frames,_=cache_inputs(12);native=[]
 for base in ('XIV','XIII'):
  digests=[]
  for residual in (False,True):
   seed_all();model,cp=initialize('jev',20261009,revision=revision,base=base)
   actor=model.eval() if residual else FrozenPlanControl(model.identity,'parent').eval()
   tracker=build_tracker(12,actor,react_learned=False);prune(tracker);ex=attach_jevtrack(tracker);commits=[];last=[None]
   def after(**d):
    commits.append((d['frame'],d['view'],d['instances'][-1].track_ids.cpu().tolist(),d['events']))
    last[0]=phase13_prefix(tracker,{k:v for k,v in d.items() if k!='events'})
   ex.commit_observer=after
   with torch.no_grad():raw,_=run(tracker,values,frames,stop=95)
   digests.append(dict(state=fingerprint(last[0]),raw=fingerprint(raw[:192]),commits=fingerprint(commits),payloads=len(commits)))
   del tracker,actor,model,ex,raw;gc.collect();torch.cuda.empty_cache()
  assert digests[0]==digests[1];native.append(dict(base=base,exact_native_parity=digests[0]))
  print('RECOVERY_NATIVE_PARITY',base,flush=True)
 result=dict(status='PASS',revision=revision,binding=binding('Actual TRAIN12 96-scene native mutated-state parity of XIV and XIII zero residual'),initializers=results,native=native,DEFER_expansion=True,capacity_conflict_globally_resolved=True,zero_K_private_DEFER=True,hard_quarantine=False,learned_memory_admission=False,legacy_XVI_memory_failure='Guard unchanged native Bank/NEW/gallery, no new memory controller; full learned-weight regression still pending')
 target=('MODEL_AND_NATIVE_CONTRACT_TESTS.json' if revision==1 else f'R{revision}_MODEL_AND_NATIVE_CONTRACT_TESTS.json') if tag is None else f'R{revision}_{tag}_CONTRACT_TESTS.json'
 save(REPORTS/target,result);print('RECOVERY_CONTRACT_PASS',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--revision',type=int,default=1);p.add_argument('--tag');a=p.parse_args()
 try:main(a.revision,a.tag)
 except Exception:failure('native_contract',traceback.format_exc());raise
