"""Direct lawful readout from typed/ordinary representations, no plan residual."""
import torch
from gtr.modeling.jev_recovery.model import RecoveryTrack
from gtr.modeling.jev_stage2.assignment import lawful_choice
class DirectAssociation(RecoveryTrack):
 def encode(self,x):
  d=super().encode(x)
  if self.arm=='attention':
   # Reuse the former joint-plan Reader as a genuine third untyped context
   # read. Its parameters remain active, matching JEV's typed-stream fusion.
   b,q,k,n=d['features'].shape
   d['features']=self.norm(self.plan_reader(d['features'].reshape(b,q*k,n),d['shared']).reshape(b,q,k,n))
   d['edge_logits']=d['base_logits']+self.edge_value(d['features']).squeeze(-1)
  return d
 def details(self,x):
  # No constrained proposal generation: one assignment on all lawful scores.
  if not x['question_mask'].any():
   empty=x['detection_visual'].new_empty(len(x['question_mask']),0,x['identity_mask'].shape[1]+1)
   return dict(base_logits=empty,edge_logits=empty,plans=[[]],plan_scores=[])
  result=self.encode(x);plans=[];scores=[]
  for i in range(len(x['question_mask'])):
   q=int(x['question_mask'][i].sum());k=int(x['identity_mask'][i].sum())
   score=torch.cat([result['edge_logits'][i,:q,:k],result['edge_logits'][i,:q,-1:]],-1)
   choice=tuple(lawful_choice(score,x['legal'][i,:q,:k]))
   plans.append([choice])
   cols=torch.tensor([k if c<0 else c for c in choice],device=score.device,dtype=torch.long)
   scores.append(score[torch.arange(q,device=score.device),cols].double().sum()[None])
  result.update(plans=plans,plan_scores=scores,arm=self.arm,direct_readout=True)
  return result
 def score_plans(self,d,x,i,alternatives):
  # Eliminate joint value from the executable path. No ignored trainable head
  # is used to inflate the active capacity comparison.
  q=int(x['question_mask'][i].sum());k=int(x['identity_mask'][i].sum())
  z=torch.cat([d['edge_logits'][i,:q,:k],d['edge_logits'][i,:q,-1:]],-1)
  return torch.stack([z[torch.arange(q,device=z.device),torch.tensor([k if c<0 else c for c in p],device=z.device)].double().sum() for p in alternatives])
