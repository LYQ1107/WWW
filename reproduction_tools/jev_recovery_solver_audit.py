"""Same real TRAIN16 state: scorer/solver attribution and common intervention budgets.

Current labels only, all methods same legal identities and eight proposals.
Forced-budget diagnostics are separate from natural policy behavior and full MOT.
"""
import argparse,collections,torch,traceback,math
from jev_recovery_common import *
from jev_recovery_learning import load_records,load_model,batch
from gtr.modeling.jev_stage2.assignment import lawful_choice
def score(r,before,after):
 c=collections.Counter();k=len(r['refs'])
 for row,(a,b) in enumerate(zip(before,after)):
  aa=k if a<0 else a;bb=k if b<0 else b;c['rows']+=1;c['changed_rows']+=a!=b
  known=bool(r['known_options'][row,bb]);c['selected_known' if known else 'selected_UNKNOWN']+=1
  if known:c['selected_correct']+=bool(r['positive'][row,bb])
  if r['known_options'][row,aa] and known:c[f"N{int(r['positive'][row,aa])}{int(r['positive'][row,bb])}"]+=1
  else:c['UNKNOWN_comparison']+=1
 return dict(c)
def main(revision,base,phase):
 protect();torch.set_num_threads(1);_,records,sources=load_records(revision)
 plans_cache={};inputs={};results=[];budgets=[];order=sorted(range(len(records)),key=lambda i:hashlib.sha256(json.dumps(records[i]['key']).encode()).hexdigest())
 for arm in ['threshold','mlp','attention','jev']:
  model,cp,_=load_model(arm,20261009,phase,revision,base);counts=collections.defaultdict(collections.Counter);events=[]
  with torch.no_grad():
   for i,r in enumerate(records):
    x,y=batch([r]);d=model.details(x);pp=d['plans'][0];base_choice=pp[0]
    input_sha=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for k,v in sorted(x.items()))).hexdigest()
    if i in plans_cache:assert pp==plans_cache[i] and input_sha==inputs[i]
    else:plans_cache[i]=pp;inputs[i]=input_sha
    original=lawful_choice(d['base_logits'][0],x['legal'][0]);repeat=lawful_choice(d['base_logits'][0],x['legal'][0]);assert original==repeat
    edge=lawful_choice(d['edge_logits'][0],x['legal'][0]);edge_repeat=lawful_choice(d['edge_logits'][0],x['legal'][0]);assert edge==edge_repeat
    selected=int(d['plan_scores'][0].argmax());choice=pp[selected]
    for name,p in [('base_single',original),('base_repeat',repeat),('edge_single',edge),('edge_repeat',edge_repeat),('joint_eight_proposals',choice)]:counts[name].update(score(r,base_choice,p))
    counts['natural_policy']['events']+=1;counts['natural_policy']['nonbase_events']+=selected!=0
    counts['natural_policy'].update(score(r,base_choice,choice))
    alternative=max(range(1,len(pp)),key=lambda j:float(d['plan_scores'][0][j])) if len(pp)>1 else None
    events.append((base_choice,pp[alternative] if alternative is not None else base_choice))
    if i%128==0:
     save(OUT/f'solver_audit_R{revision}_{base}/PROGRESS.json',dict(status='RUNNING',arm=arm,record=i,total=len(records)));print('SOLVER_AUDIT',arm,i,len(records),flush=True)
  eligible=[i for i in order if len(plans_cache[i])>1]
  for rate in [0.,.01,.05,.10,.25,1.]:
   number=math.floor(len(eligible)*rate);selected_events=set(eligible[:number]);c=collections.Counter()
   for i,r in enumerate(records):
    a,b=events[i];c.update(score(r,a,b if i in selected_events else a))
   budgets.append(dict(arm=arm,rate=rate,eligible_events=len(eligible),exact_intervened_events=number,counts=dict(c),GT_selected_events=False,forced_best_nonbase_not_natural_policy=True))
  results.append(dict(arm=arm,checkpoint=cp,factors={k:dict(v) for k,v in counts.items()}))
  del model;torch.cuda.empty_cache()
 value=dict(status='COMPLETE_CURRENT_STATE_DIAGNOSTIC',binding=binding('Shared native TRAIN16 tensors, legal action constraints and GT-free proposals'),revision=revision,base=base,phase=phase,cases=results,common_budget_diagnostics=budgets,sources=sources,exact_inputs_and_proposals_equal=True,unchanged_solver_repeat_verified=True,edge_Hungarian_scope='Full additive legal pool; it cannot optimize non-additive joint value. Compare scorer/solver factors separately from the same-eight-plan joint comparison.',budget_scope='Identical SHA event order, no GT/confidence event selection; forced nonbase diagnostics do not replace original natural runs or full mutated-state MOT metrics',GT_actor_input=False,independent_validation=False,future_safety=False)
 save(REPORTS/f'R{revision}_{base}/SOLVER_AND_BUDGET_ATTRIBUTION_{phase}.json',value);print('SOLVER_AUDIT_COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--revision',type=int,default=2);p.add_argument('--base',default='XIV');p.add_argument('--phase',default='pilot');a=p.parse_args()
 try:main(a.revision,a.base,a.phase)
 except Exception:failure('same_solver_budget_audit',traceback.format_exc());raise
