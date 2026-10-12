"""Controlled residual adapters; zero residual keeps the exact strong base output."""
import torch
from torch import nn
from gtr.modeling.jev_oneweek.model import JevTrack
from gtr.modeling.jev_stage2.model import GlobalIdentityJev
from .plans import plans

class FixedBase(nn.Module):
 def __init__(self):super().__init__();self.core=GlobalIdentityJev('fixed_question')
 def details(self,x):return dict(logits=self.core(x))
 def forward(self,x):return self.core(x)

class RecoveryTrack(JevTrack):
 def __init__(self,arm='jev',revision=1,base='XIV',maximum=4):
  super().__init__(arm);assert revision in (1,2,3) and base in ('XIV','XIII')
  self.revision=revision;self.base=base;self.maximum=maximum
  if base=='XIII':
   self.identity=FixedBase()
   for p in self.identity.parameters():p.requires_grad_(False)
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
