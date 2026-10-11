"""Two preregistered positive-weight views; native raw state and Q4 stay intact."""
import torch
import torch.nn.functional as F
from .factor_memory import FactorMemory
from gtr.modeling.jev_phase15.native_commit_adapter import CommitmentMemory

class SoftEvidenceMemory(FactorMemory):
 def __init__(self,rule,prefix=None):
  assert rule in ['E1','E2'];super().__init__('B0',prefix);self.soft_rule=rule
 def build(self,current,refs,galleries,frame,view,active,**kwargs):
  raw=CommitmentMemory.build(self,current,refs,galleries,frame,view,active,**kwargs)
  x={k:v.clone() for k,v in raw.items()}
  for col,identity in enumerate(refs):
   g=galleries[identity];records=self.admission.records.get(identity,[])
   # At cold bootstrap no qualified per-view metadata exists. Preserve baseline.
   if not records:continue
   assert len(records)==len(g)
   features=g.reid_features[:,:1024]
   cams=torch.tensor([r['view'] for r in records],device=features.device)
   age=features.new_tensor([max(0,frame-r['frame']) for r in records])
   weights=.25+.75*torch.exp(-age/40) if self.soft_rule=='E1' else torch.ones_like(age)
   def mean(mask):
    w=weights[mask];v=features[mask];return F.normalize((v*w[:,None]).sum(0)/w.sum(),dim=-1)
   global_value=mean(torch.ones(len(records),dtype=torch.bool,device=features.device))
   if self.soft_rule=='E2':
    # Half raw global mean, half equal-camera means; no discarded observations.
    camera_means=[features[cams==c].mean(0) for c in cams.unique()]
    global_value=F.normalize(.5*features.mean(0)+.5*torch.stack(camera_means).mean(0),dim=-1)
   x['history_visual'][0,col,1]=global_value
   if self.soft_rule=='E1':
    for slot,c in [(2,view),(3,1-view)]:
     mask=cams==c
     if bool(mask.any()) and bool(raw['history_mask'][0,col,slot]):x['history_visual'][0,col,slot]=mean(mask)
  x['pair_evidence'][:,:,:,:4]=torch.einsum('bqd,bkhd->bqkh',x['detection_visual'],x['history_visual'])
  for key in ['history_mask','identity_meta','commitment_features','legal']:
   assert torch.equal(x[key],raw[key]),key
  assert torch.equal(x['history_visual'][:,:,0],raw['history_visual'][:,:,0])
  self.last_comparison={'raw':raw,'safe':x,'refs':list(refs),'key':(frame,view)}
  return x
