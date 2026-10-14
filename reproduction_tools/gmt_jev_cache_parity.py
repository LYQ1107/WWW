"""Full native Rule score/feature pairing: optional cache versus uncached math."""
import gzip,torch
from gmt_jev_runtime import *

def main():
 from gtr.modeling.gmt_jev import IdentityCorrection
 from run_jev_phase10_closed_loop import raw_predictions
 values,frames,images=metadata(16);seed();m,_,_=model()
 old=IdentityCorrection('rule',cache_features=False).cuda().eval()
 new=IdentityCorrection('rule',cache_features=True).cuda().eval();calls=[0]
 class Paired(torch.nn.Module):
  enabled=True
  def __init__(self):super().__init__();self.old=old;self.new=new
  def forward(self,*a,**kw):
   before=old(*a,**kw);after=new(*a,**kw)
   for key in ['features','eligible','gate','residual','corrected_scores']:
    assert torch.equal(old.last[key],new.last[key]),(kw,key)
   assert torch.equal(before,after);calls[0]+=1;return after
 m.gmt_identity_correction=Paired()
 def committed(**d):
  inst=d['instances'][-1];v,f=d['view'],d['frame'];old.committed(inst,v,f);new.committed(inst,v,f)
 m.jev_candidate_commit_observer=committed;seed()
 with torch.no_grad():raw,_=run(m,values,frames)
 pred=raw_predictions(raw,images);reference=read(REPORTS/'ONLINE_rule.json')['raw_predictions']
 assert sha(reference['path'])==reference['SHA256']
 with gzip.open(reference['path'],'rt') as f:previous=json.load(f)
 assert pred==previous and calls[0]==2*frames-1
 result=dict(status='PASS_EXACT_FULL_NATIVE_RULE',frames=frames,video=16,score_feature_pairs=calls[0],
  exact_fields=['features','eligible','gate','residual','corrected_scores','complete_native_predictions'],
  reference=reference,no_GT_inputs=True,no_gradient_updates=True,no_actor_parameter_changes=True,
  cache='Only normalized committed per-camera sums and history consistency; recompute on actual committed observations. No candidate/solver/threshold/ID/Gallery/Bank policy change.',
  proof_scope='Both policies receive the identical evolving native states and commits; new full raw predictions also match the retained original uncached Rule run.',source=binding())
 save(REPORTS/'FEATURE_CACHE_PARITY.json',result);print('GMT_FEATURE_CACHE_PARITY',result['status'],calls[0],flush=True)

if __name__=='__main__':main()
