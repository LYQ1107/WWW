"""Controlled residual adapters; zero residual keeps the exact strong base output."""
import torch
from torch import nn
from gtr.modeling.jev_oneweek.model import JevTrack
from gtr.modeling.jev_stage2.model import GlobalIdentityJev
from .plans import plans
from gtr.modeling.jev_oneweek.plans import selected_tokens

class FixedBase(nn.Module):
 def __init__(self):super().__init__();self.core=GlobalIdentityJev('fixed_question')
 def details(self,x):return dict(logits=self.core(x))
 def forward(self,x):return self.core(x)

class RecoveryTrack(JevTrack):
 def __init__(self,arm='jev',revision=1,base='XIV',maximum=4):
  original_arm=arm
  mapped={'f0_mean':'jev','mlp_fusion':'jev','attention_fusion':'jev','no_interaction':'jev'}.get(arm,arm)
  super().__init__(mapped);assert revision in (1,2,3) and base in ('XIV','XIII')
  self.recovery_arm=original_arm
  self.revision=revision;self.base=base;self.maximum=maximum
  if base=='XIII':
   self.identity=FixedBase()
   for p in self.identity.parameters():p.requires_grad_(False)
  if revision>=2 and original_arm=='mlp_fusion':
   del self.plan_reader
   self.fusion=nn.Sequential(nn.Linear(4*self.d,104),nn.GELU(),nn.Linear(104,self.d))
  if revision>=2 and original_arm=='mlp':
   # Pure ordinary MLP sees the same full evidence; match the ~169k new budget.
   self.ordinary=nn.Sequential(nn.Linear(5*self.d,200),nn.GELU(),nn.Linear(200,self.d))
  if revision>=2:self.early_evidence=nn.Linear(1,self.d)

 def encode(self,x):
  if self.revision==1:return super().encode(x)
  plain={key:value for key,value in x.items() if key not in ('anchor_visual','anchor_mask')}
  k=x['identity_mask'].shape[1]
  if 'anchor_visual' in x:
   cosine=torch.einsum('bqd,bkd->bqk',x['detection_visual'],x['anchor_visual'])
   extra=self.early_evidence(cosine[...,None])*x['anchor_mask'][:,None,:,None]
  else:extra=x['detection_visual'].new_zeros(len(x['question_mask']),x['question_mask'].shape[1],k,self.d)
  # Identical additional causal evidence reaches JEV and ordinary controls
  # before their nonlinear processing. It never changes the frozen base input.
  old_evidence=self.evidence.forward
  self.evidence.forward=lambda value:old_evidence(value)+extra
  captured={};hooks=[];original_state=self.identity.core.encode_state
  def state(value):
   out=original_state(value);captured['history']=out[1];return out
  self.identity.core.encode_state=state
  for name,head in [('cross',self.cross_head),('stability',self.continuity_head)]:
   hooks.append(head.register_forward_pre_hook(lambda m,a,n=name:captured.update({n:a[0]})))
  old_readers=[]
  # One actual frozen State encode. Each typed question receives its own
  # identity-specific historical view in addition to the shared slots.
  if hasattr(self,'readers') and self.arm!='attention':
   for task,reader in enumerate(self.readers):
    old=reader.forward;old_readers.append((reader,old))
    def typed(query,shared,mask=None,task=task,old=old):
     b,q=x['question_mask'].shape;k=x['identity_mask'].shape[1];d=self.d
     view=3 if task==0 else 2
     h=self.history(captured['history'][:,:,view])*x['history_mask'][:,:,view,None]
     h=torch.cat([h,h.new_zeros(b,1,d)],1)[:,None].expand(-1,q,-1,-1)
     keys=torch.cat([shared[:,None,None].expand(-1,q,k+1,-1,-1),h[:,:,:,None]],3)
     return old(query.reshape(b*q*(k+1),1,d),keys.reshape(b*q*(k+1),-1,d)).reshape(b,q*(k+1),d)
    reader.forward=typed
  try:result=super().encode(plain)
  finally:
   self.evidence.forward=old_evidence
   self.identity.core.encode_state=original_state
   for hook in hooks:hook.remove()
   for reader,old in old_readers:reader.forward=old
  cross=captured['cross'];stable=captured['stability'];b,q,kplus,d=cross.shape
  # F0 retains the exact mean operation. F1 retains two streams through an MLP.
  # F2 reads separately typed streams for each concrete legal identity action.
  if self.recovery_arm=='mlp_fusion':
   feature=self.norm(self.fusion(torch.cat([cross,stable,cross-stable,cross*stable],-1)))
  elif self.recovery_arm in ('jev','fixed_question','cross_only','stability_only','no_shared','no_question_reader','attention_fusion'):
   keys=torch.stack([cross,stable],-2)
   if self.recovery_arm!='attention_fusion':keys=keys+self.tasks.weight[None,None,None,:,:]
   query=result['shared'].mean(1)[:,None,None].expand(-1,q,kplus,-1)
   feature=self.norm(self.plan_reader(query.reshape(b*q*kplus,1,d),keys.reshape(b*q*kplus,2,d)).reshape(b,q,kplus,d))
  else:feature=(cross+stable)/2
  result['features']=feature;result['cross_features']=cross;result['stability_features']=stable
  if self.arm in ('threshold','binary'):
   delta=-self.edge_value(feature[:,:,:-1]).squeeze(-1);delta=torch.cat([delta,delta.new_zeros(b,q,1)],-1)
  else:delta=self.edge_value(feature).squeeze(-1)
  result['edge_logits']=result['base_logits']+delta
  return result

 def score_plans(self,details,x,index,alternatives):
  if self.revision==1 or self.recovery_arm not in ('mlp_fusion','no_interaction'):
   return super().score_plans(details,x,index,alternatives)
  q=int(x['question_mask'][index].sum());k=int(x['identity_mask'][index].sum())
  score=torch.cat([details['edge_logits'][index,:q,:k],details['edge_logits'][index,:q,-1:]],-1)
  tokens=torch.cat([details['features'][index,:q,:k],details['features'][index,:q,-1:]],1)
  selected,indices=selected_tokens(tokens,alternatives);rows=torch.arange(q,device=score.device)[None].expand(len(alternatives),-1)
  initial=score.double()[rows,indices].sum(-1)
  if self.recovery_arm=='no_interaction':return initial
  mean=selected.mean(1);maximum=selected.max(1).values
  residual=self.plan_value(self.fusion(torch.cat([mean,maximum,mean-maximum,mean*maximum],-1))).squeeze(-1)
  return initial+residual.double()
 def details(self,x):
  if not x['question_mask'].any():
   k=x['identity_mask'].shape[1];empty=x['detection_visual'].new_empty(len(x['question_mask']),0,k+1)
   return dict(base_logits=empty,edge_logits=empty,plans=[[]],plan_scores=[])
  result=self.encode(x);all_plans=[];scores=[]
  for index in range(len(x['question_mask'])):
   q=int(x['question_mask'][index].sum());k=int(x['identity_mask'][index].sum())
   logits=torch.cat([result['base_logits'][index,:q,:k],result['base_logits'][index,:q,-1:]],-1)
   pp=plans(logits,x['legal'][index,:q,:k],maximum=self.maximum)
   all_plans.append(pp);scores.append(self.score_plans(result,x,index,pp))
  result.update(plans=all_plans,plan_scores=scores,arm=self.arm)
  return result
