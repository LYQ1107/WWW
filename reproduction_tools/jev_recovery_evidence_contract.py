"""Training/native early-prototype equality on real source trajectory; both base parity."""
import gc,torch,traceback
from jev_recovery_common import *
from jev_recovery_learning import initialize,load_records,batch
from jev_recovery_online import configure
from jev_phase14_artifacts import load_dense
def main():
 protect();torch.set_num_threads(1)
 from jev_phase13_runtime import cache_inputs,build_tracker,run
 from jev_phase14_latency import prune
 from gtr.modeling.jev_recovery.native import attach_jevtrack
 from gtr.modeling.jev_native_state import fingerprint
 from audit_jev_stage2_gta_free import phase13_prefix
 import jev_oneweek_online as online
 configure(online,2,'XIV')
 cp=read(REPORTS/'R2_EARLY_EVIDENCE.json')['patch'];assert sha(cp['path'])==cp['SHA256']
 patch={tuple(r['key']):r for r in load_dense(cp['path'])['patches']}
 values,frames,_=cache_inputs(12);cases=[];maxerr=0.;tested=0
 for base in ('XIV','XIII'):
  pairs=[]
  for new in (False,True):
   if new:actor,checkpoint=initialize('jev',20261009,2,base);actor.eval()
   else:actor,checkpoint=configure(online,2,base).policy('parent',20261009,'pilot')
   tracker=build_tracker(12,actor,react_learned=False);prune(tracker);ex=attach_jevtrack(tracker);last=[None];events=[]
   def before(**d):
    nonlocal maxerr,tested
    if base!='XIV' or d['task']!=0:return
    key=(12,d['context']['frame'],d['context']['view']);expected=patch[key]
    value=d['batch']['anchor_visual'][0].cpu();mask=d['batch']['anchor_mask'][0].cpu()
    assert torch.equal(mask,expected['anchor_mask'])
    error=float((value-expected['anchor_visual']).abs().max()) if value.numel() else 0.
    maxerr=max(maxerr,error);tested+=1;assert error<=2e-6
   def after(**d):
    events.append((d['frame'],d['view'],d['instances'][-1].track_ids.cpu().tolist(),d['events']))
    last[0]=phase13_prefix(tracker,{k:v for k,v in d.items() if k!='events'})
   ex.observer=before;ex.commit_observer=after
   with torch.no_grad():raw,_=run(tracker,values,frames,stop=95)
   pairs.append(dict(state=fingerprint(last[0]),raw=fingerprint(raw[:192]),commits=fingerprint(events)))
   del raw,actor,tracker,ex;gc.collect();torch.cuda.empty_cache()
  assert pairs[0]==pairs[1];cases.append(dict(base=base,zero_residual_native_exact=True,sha=pairs[0]))
 train,val,_=load_records(2);records=[train[0],train[len(train)//2],train[-1]];x,y=batch(records);model,_=initialize('jev',20261009,2);model.eval()
 with torch.no_grad():d=model.details(x);expected=model.identity({k:v for k,v in x.items() if k not in ('commitment_features','anchor_visual','anchor_mask')})
 assert torch.equal(d['edge_logits'],expected)
 # Every ordinary controller receives the same prototype tensors and common
 # learned projection, with its own scoring architecture.
 param=[]
 for arm in ('jev','f0_mean','mlp_fusion','attention_fusion','mlp','attention','threshold'):
  other,_=initialize(arm,20261009,2);assert torch.equal(model.early_evidence.weight,other.early_evidence.weight)
  param.append(dict(arm=arm,new_parameters=sum(p.numel() for p in other.parameters() if p.requires_grad)))
 save(REPORTS/'R2_EARLY_EVIDENCE_CONTRACT.json',dict(status='PASS',binding=binding('Actual training/native earliest appearance evidence and unchanged base outputs'),cases=cases,actual_source_payloads_tested=tested,max_abs_training_native_prototype_error=maxerr,all_controls_same_common_evidence_projection=True,parameters=param,early_cache_reset_with_native_reset=True,no_extra_perception=True,GT_in_actor=False))
 print('EARLY_EVIDENCE_CONTRACT_PASS',tested,maxerr,flush=True)
if __name__=='__main__':
 try:main()
 except Exception:failure('early_evidence_native_contract',traceback.format_exc());raise
