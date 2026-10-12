"""Certified typed-answer evaluation and actual question/shared-compute interventions."""
import argparse,collections,torch,traceback
from jev_recovery_common import *
from jev_recovery_learning import load_records,load_model,batch
def main(revision,base,phase):
 protect();torch.set_num_threads(1);_,records,sources=load_records(revision)
 counts={};cases=[]
 for arm in ['jev','threshold','mlp','attention']:
  actor,cp,_=load_model(arm,20261009,phase,revision,base);stats=collections.Counter()
  with torch.no_grad():
   for r in records:
    a=r['oneweek_labels']['cross'];b=r['oneweek_labels']['stability'];mask=(a>=0)&(b>=0)&(a!=b)
    if not mask.any():continue
    x,y=batch([r]);d=actor.details(x);p=d['cross_logits'][0]>=0;q=d['stability_logits'][0]>=0;mm=mask.to(p.device);aa=a.to(p.device);bb=b.to(q.device)
    stats['target_disagreements']+=int(mm.sum());stats['events']+=1
    stats['cross_correct']+=int(((p==aa)&mm).sum());stats['persistent_correct']+=int(((q==bb)&mm).sum());stats['both_correct']+=int(((p==aa)&(q==bb)&mm).sum())
  cases.append(dict(arm=arm,checkpoint=cp,counts=dict(stats)))
  del actor;torch.cuda.empty_cache()
 actor,cp,_=load_model('jev',20261009,phase,revision,base);r=next(r for r in records if len(r['rows'])>=2 and len(r['refs'])>=4);x,y=batch([r]);n=[0]
 original=actor.identity.core.encode_state
 def encoded(value):n[0]+=1;return original(value)
 actor.identity.core.encode_state=encoded
 with torch.no_grad():
  normal=actor.details(x);one=n[0]
  weight=actor.tasks.weight.clone();actor.tasks.weight.copy_(weight.flip(0));swapped=actor.details(x);actor.tasks.weight.copy_(weight)
  sensitivity=float((normal['cross_logits']-swapped['cross_logits']).abs().mean())
  q=len(r['rows']);pp=[]
  for a,b in [(0,2),(1,3),(0,3),(1,2)]:
   p=[-1]*q;p[0]=a;p[1]=b;pp.append(tuple(p))
  values=actor.score_plans(normal,x,0,pp);interaction=float(values[0]+values[1]-values[2]-values[3])
  arm=actor.arm;actor.arm='no_shared';n[0]=0;repeated=actor.details(x);double=n[0];actor.arm=arm
 assert one==1 and double==2
 assert torch.equal(normal['edge_logits'],repeated['edge_logits'])
 value=dict(status='COMPLETE',binding=binding('Actual independently different TRAIN16 targets and trained functional interventions'),revision=revision,base=base,phase=phase,certified_target_disagreement_cases=cases,actual_State_encodes_single=one,actual_State_encodes_recomputed=double,recomputed_edge_logits_bitwise_equal=True,swapped_question_cross_output_mean_abs_change=sensitivity,legal_four_plan_nonadditive_interaction=interaction,checkpoint=cp,sources=sources,examples=[list(p) for p in pp],functional_components_not_proof_unique_tracking_gain=True,independent_validation=False)
 save(REPORTS/f'R{revision}_{base}/QUESTION_MECHANISM_{phase}.json',value);print('QUESTION_MECHANISM_COMPLETE',revision,base,interaction,sensitivity,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--revision',type=int,default=2);p.add_argument('--base',default='XIV');p.add_argument('--phase',default='pilot');a=p.parse_args()
 try:main(a.revision,a.base,a.phase)
 except Exception:failure('question_mechanism_probe',traceback.format_exc());raise
