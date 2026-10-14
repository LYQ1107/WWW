"""Small post-GTA residual. No GT, alternate solver or independent Gallery."""
import math
import torch
from torch import nn
from torch.nn import functional as F

FEATURE_NAMES=['GMT_average','row_top1','row_top2','row_margin','entropy','candidate_count',
 'window_support','committed_support','age','same_camera_support','other_camera_support',
 'same_camera_cosine','other_camera_cosine','other_minus_same','camera_conflict',
 'history_consistency','detection_confidence','raw_score_scaled','other_camera_present',
 'top_GMT_support','same_camera_present','threshold_distance','first_camera_link','both_cameras_present']

class ResidualHead(nn.Module):
 def __init__(self,kind):
  super().__init__();assert kind in ['mlp','attention','cic'];self.kind=kind
  if kind=='mlp':
   self.hidden=nn.Sequential(nn.Linear(72,43),nn.GELU(),nn.Linear(43,43),nn.GELU())
   self.gate=nn.Linear(43,1);self.delta=nn.Linear(43,1)
  else:
   self.edge=nn.Linear(24,32);self.query=nn.Linear(24,32)
   self.key=nn.Linear(32,32);self.value=nn.Linear(32,32)
   self.mix=nn.Linear(32,32);self.norm=nn.LayerNorm(32)
   if kind=='cic':self.question_type=nn.Embedding(2,32)
   self.gate=nn.Linear(32,1);self.delta=nn.Linear(32,1)
  # Zero residual with a nonzero sigmoid gate: the output head has gradients.
  nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias)
  nn.init.zeros_(self.gate.weight);nn.init.zeros_(self.gate.bias)
 def forward(self,x):
  if not x.shape[0] or not x.shape[1]:return x.new_zeros(x.shape[:2]),x.new_zeros(x.shape[:2])
  if self.kind=='mlp':
   glob=x.mean((0,1),keepdim=True).expand_as(x);row=x.mean(1,keepdim=True).expand_as(x)
   h=self.hidden(torch.cat([x,glob,row],-1))
  else:
   edge=F.gelu(self.edge(x));tokens=edge.mean(0);query=self.query(x.mean(1))
   weights=torch.softmax(query@self.key(tokens).T/math.sqrt(32),-1)
   context=weights@self.value(tokens)
   h=edge+query[:,None]+context[:,None]
   if self.kind=='cic':h=h+self.question_type((x[...,13]>.05).long())
   h=F.gelu(self.mix(self.norm(h)))
  return torch.sigmoid(self.gate(h)).squeeze(-1),torch.tanh(self.delta(h)).squeeze(-1)

class IdentityCorrection(nn.Module):
 def __init__(self,kind='observe',enabled=True):
  super().__init__();assert kind in ['observe','threshold','rule','confidence','mlp','attention','cic']
  self.kind=kind;self.enabled=enabled
  self.head=ResidualHead(kind) if kind in ['mlp','attention','cic'] else None
  self.observer=None;self.reset()
 def reset(self):self.summaries={};self.last=None
 def _add(self,identity,feature,view,frame):
  vector=F.normalize(feature[:1024].detach(),dim=0)
  state=self.summaries.setdefault(int(identity),dict(sums={},counts={},last=-1))
  state['sums'][view]=state['sums'].get(view,torch.zeros_like(vector))+vector
  state['counts'][view]=state['counts'].get(view,0)+1;state['last']=frame
 def committed(self,instance,view,frame):
  if self.kind=='confidence' and self.observer is None:return
  for identity,feature in zip(instance.track_ids.tolist(),instance.reid_features):self._add(identity,feature,view,frame)
 def features(self,scores,ids,lengths,current,galleries,view,frame,first):
  q,k=scores.shape
  if first:
   for t in ids.tolist():
    if t not in self.summaries and t in galleries:
     # The native first-frame bootstrap is already committed in the other camera.
     assert len(galleries[t])==1
     self._add(t,galleries[t].reid_features[0],1-view,frame)
  if not q or not k:return scores.new_zeros((q,k,24)),scores.new_zeros((q,k),dtype=torch.bool)
  avg=scores/lengths.clamp_min(1)[None];top=avg.topk(min(2,k),dim=1).values
  top1=top[:,0];top2=top[:,1] if k>1 else torch.zeros_like(top1);margin=top1-top2
  probs=torch.softmax(avg*10,-1);entropy=-(probs*probs.clamp_min(1e-8).log()).sum(-1)/max(1.,math.log(k))
  same=[];other=[];ns=[];no=[];total=[];age=[];consistency=[]
  zero=scores.new_zeros(1024)
  for t in ids.tolist():
   s=self.summaries.get(t,dict(sums={},counts={},last=-1));a=s['sums'].get(view,zero);b=s['sums'].get(1-view,zero)
   ca=s['counts'].get(view,0);cb=s['counts'].get(1-view,0)
   same.append(F.normalize(a,dim=0));other.append(F.normalize(b,dim=0));ns.append(ca);no.append(cb);total.append(ca+cb)
   age.append(max(0,frame-s['last']));consistency.append((a+b).norm()/max(1,ca+cb))
  obs=F.normalize(current.reid_features[:,:1024].detach(),dim=-1)
  same_cos=obs@torch.stack(same).T;other_cos=obs@torch.stack(other).T
  ns=scores.new_tensor(ns);no=scores.new_tensor(no);total=scores.new_tensor(total);age=scores.new_tensor(age)
  confidence=current.scores if current.has('scores') else current.objectness_logits
  col=lambda v:v[None].expand(q,k);row=lambda v:v[:,None].expand(q,k)
  x=torch.stack([avg,row(top1),row(top2),row(margin),row(entropy),torch.full_like(avg,math.log1p(k)/5),
   col(lengths.log1p()/8),col(total.log1p()/10),col(age.clamp_max(400)/400),col(ns.log1p()/10),col(no.log1p()/10),
   same_cos,other_cos,other_cos-same_cos,(other_cos-same_cos).abs(),col(torch.stack(consistency)),row(confidence),scores.clamp_max(100)/100,
   col((no>0).float()),(avg==top1[:,None]).float(),col((ns>0).float()),avg-.1,torch.full_like(avg,float(first)),col(((ns>0)&(no>0)).float())],-1)
  # A frozen observable risk gate; no labels/GT purity enter deployment.
  eligible=(margin[:,None]<=.10)&(no[None]>=3)&((other_cos-same_cos).abs()>=.05)
  if k<2:eligible.zero_()
  return x,eligible
 def forward(self,scores,ids,lengths,current,galleries,*,view,frame,first):
  if not self.enabled:return scores
  if self.kind=='confidence' and self.observer is None:
   average=scores/lengths.clamp_min(1)[None]
   return scores-(.1-average).clamp(0,.05)*lengths[None]
  x,eligible=self.features(scores,ids,lengths,current,galleries,view,frame,first)
  if self.kind=='observe':new=scores;gate=torch.zeros_like(scores);residual=torch.zeros_like(scores)
  elif self.kind=='threshold':
   gate=eligible.float();residual=-.02*gate;new=scores+residual*lengths[None]
  elif self.kind=='rule':
   gate=eligible.float();residual=.05*gate*(x[...,13]/.2).clamp(-1,1);new=scores+residual*lengths[None]
  elif self.kind=='confidence':
   eligible=x[...,0]<.1;gate=eligible.float();residual=-(.1-x[...,0]).clamp(0,.05);new=scores+residual*lengths[None]
  else:
   gate,delta=self.head(x);gate=gate*eligible.float();residual=.05*gate*delta;new=scores+residual*lengths[None]
  assert torch.isfinite(new).all() and torch.equal(new[~eligible],scores[~eligible])
  self.last=dict(features=x,eligible=eligible,base_scores=scores,corrected_scores=new,ids=ids,lengths=lengths,gate=gate,residual=residual,view=view,frame=frame,first=first)
  if self.observer is not None:self.observer(self.last,current,galleries)
  return new
