"""GT-free candidate expansion; private DEFER and per-camera capacity remain native."""
import torch
from gtr.modeling.jev_stage2.assignment import lawful_choice
from gtr.modeling.jev_oneweek.plans import plans as old_plans

def plans(logits,legal,maximum=4,mode='expanded',affinity=None):
 assert maximum in (4,8) and mode in ('old','expanded')
 old=old_plans(logits,legal)
 if mode=='old':return old
 result=[old[0]];q,k=legal.shape
 if not q or not k:return result
 z=logits.detach().double();base=result[0];trials=[]
 # P1: any legal existing identity may compete with a currently deferred row.
 # P2/P3: force a scored rival, re-solving *all* rows so capacity changes are real.
 for row in range(q):
  cols=torch.where(legal[row])[0].tolist()
  cols.sort(key=lambda c:(-float(z[row,c]),c))
  for col in cols[:2]:
   if col==base[row]:continue
   gap=float(z[row,base[row] if base[row]>=0 else k]-z[row,col])
   if base[row]<0 or gap<=.5:trials.append((0 if base[row]<0 else 1,gap,row,col))
  if affinity is not None:
   rival=sorted(cols,key=lambda c:(-float(affinity[row,c]),c))[:2]
   for col in rival:
    if col!=base[row] and float(affinity[row,col])>=.65:
     # Early appearance offers a bounded alternative even when a corrupted
     # mean makes the frozen scorer confidently prefer the wrong identity.
     trials.append((-1,-float(affinity[row,col]),row,col))
 trials.sort()
 # Budget four favors formerly unreachable DEFER alternatives. Budget eight
 # preserves all original proposals first. Both are explicitly separate arms.
 if maximum==8:result.extend(old[1:])
 calls=0
 for _,gap,row,col in trials:
  if len(result)>=maximum or calls>=32:break
  mask=legal.clone();mask[:,col]=False;mask[row,:]=False;mask[row,col]=True
  forced=z.clone();forced[row,-1]=min(float(z.min())-1e6,-1e6)
  candidate=tuple(lawful_choice(forced,mask));calls+=1
  assert candidate[row]==col
  if candidate not in result:result.append(candidate)
 for p in old[1:]:
  if len(result)>=maximum:break
  if p not in result:result.append(p)
 for p in result:
  existing=[c for c in p if c>=0]
  assert len(existing)==len(set(existing))
  assert all(c<0 or bool(legal[r,c]) for r,c in enumerate(p))
 return result
