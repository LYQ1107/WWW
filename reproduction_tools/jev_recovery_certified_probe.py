"""Exact same-state certified corrections; changing to UNKNOWN is not an improvement."""
import argparse,collections,torch,traceback
from jev_recovery_common import *
from jev_recovery_learning import load_model,load_records,batch
def main(arm,revision,base,phase):
 protect();torch.set_num_threads(1);_,records,sources=load_records(revision);model,checkpoint,_=load_model(arm,20261009,phase,revision,base)
 selected=[records[i] for i in torch.linspace(0,len(records)-1,min(128,len(records))).long().tolist()]
 count=collections.Counter();rows=[]
 with torch.no_grad():
  for start in range(0,len(selected),4):
   part=selected[start:start+4];x,y=batch(part);d=model.details(x)
   for i,r in enumerate(part):
    pp=d['plans'][i];before=pp[0];after=pp[int(d['plan_scores'][i].argmax())];k=len(r['refs'])
    for row,(u,v) in enumerate(zip(before,after)):
     a=k if u<0 else u;b=k if v<0 else v
     ka,kb=bool(r['known_options'][row,a]),bool(r['known_options'][row,b]);count['rows']+=1;count['changed_rows']+=u!=v
     if ka and kb:
      aa,bb=int(r['positive'][row,a]),int(r['positive'][row,b]);count[f'N{aa}{bb}']+=1
     else:
      count['UNKNOWN_comparison_rows']+=1
      if ka and not kb:count['known_to_UNKNOWN']+=1;count['known_wrong_to_UNKNOWN']+=not bool(r['positive'][row,a])
      if not ka and kb:count['UNKNOWN_to_known']+=1;count['UNKNOWN_to_known_correct']+=bool(r['positive'][row,b])
     if u!=v:rows.append(dict(key=r['key'],row=row,before=u,after=v,before_known=ka,after_known=kb,before_correct=bool(r['positive'][row,a]) if ka else None,after_correct=bool(r['positive'][row,b]) if kb else None))
 result=dict(status='COMPLETE_PAIRED_SAME_STATE_CURRENT_ASSOCIATION',binding=binding('Fixed deterministic TRAIN16 128 payload current certificate audit'),revision=revision,base=base,arm=arm,phase=phase,checkpoint=checkpoint,counts=dict(count),certified_net_corrections=count['N01']-count['N10'],sources=sources,changed_rows=rows,unknown_abstention_not_certified_correction=True,full_online=False,future_safety=False,independent_validation=False)
 save(REPORTS/f'CERTIFIED_PROBE_R{revision}_{base}_{arm}_{phase}.json',result);print('CERTIFIED_PROBE',arm,dict(count),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--revision',type=int,default=1);p.add_argument('--base',default='XIV');p.add_argument('--phase',default='pilot');a=p.parse_args()
 try:main(a.arm,a.revision,a.base,a.phase)
 except Exception:failure('certified_same_state_probe',traceback.format_exc());raise
