"""Read first three real past Gallery appearances, without changing any write rule."""
import torch
import torch.nn.functional as F
from gtr.modeling.jev_oneweek.native import attach_jevtrack as original_attach

def attach_jevtrack(model):
 ex=original_attach(model);old=ex.batch;cache={};original_reset=ex.reset
 def reset():
  cache.clear();return original_reset()
 ex.reset=reset
 def batch(current,refs,galleries,frame,view,active):
  x=old(current,refs,galleries,frame,view,active)
  proto=current.reid_features.new_zeros(1,len(refs),1024);mask=torch.zeros(1,len(refs),dtype=torch.bool,device=proto.device)
  for col,ref in enumerate(refs):
   n=min(3,len(galleries[ref]))
   if n:
    if ref not in cache or cache[ref][0]!=n:
     features=galleries[ref].reid_features[:n,:1024]
     cache[ref]=(n,F.normalize(F.normalize(features,dim=-1).mean(0),dim=-1).detach())
    proto[0,col]=cache[ref][1];mask[0,col]=True
  x['anchor_visual']=proto;x['anchor_mask']=mask
  return x
 ex.batch=batch
 return ex
