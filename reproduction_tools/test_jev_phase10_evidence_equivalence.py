"""Adversarial numeric/permutation checks for the canonical vectorized encoder."""
import json,torch
from jev_phase10_common import *
from gtr.modeling.jev_candidate_features import evidence12,evidence12_reference

def main():
 checks=[]
 for device in ['cpu','cuda:0']:
  torch.manual_seed(20261008)
  for m,n in [(0,0),(3,0),(0,9),(1,1),(8,65),(23,129)]:
   scores=torch.randn(m,n,device=device);ids=tuple(range(1,n+1));pairs={r:r%n for r in range(min(m,n))} if n else {}
   if m>1 and n>1:scores[0,0]=float('nan');scores[1,1]=float('inf')
   lengths=torch.arange(n,device=device,dtype=torch.float32)+1
   data={'hits':{t:t+8 for t in ids},'memory_lengths':{t:t+2 for t in ids},'view_fractions':{t:.5 for t in ids},'bank_eligible':ids[::2],'galleries':{t:torch.randn(512,device=device) for t in ids if t%3},'observations':torch.randn(m,512,device=device)}
   a=evidence12_reference(scores,ids,pairs,lengths,**data);b=evidence12(scores,ids,pairs,lengths,**data)
   delta=float((a-b).abs().max()) if a.numel() else 0.;assert delta<=1e-6,(device,m,n,delta)
   if n:
    perm=torch.randperm(n,device=device);inverse=torch.argsort(perm)
    rearranged=evidence12(scores[:,perm],tuple(ids[int(c)]for c in perm.cpu()),{r:int(inverse[c])for r,c in pairs.items()},lengths[perm],**data)
    permutation_error=float((b[:,perm]-rearranged).abs().max())if b.numel()else 0.
    assert permutation_error<=1e-6,(device,m,n,permutation_error)
   else:permutation_error=0.
   checks.append({'device':device,'rows':m,'candidates':n,'max_abs_error':delta,'permutation_error':permutation_error})
 result={'status':'PASS','binding':binding(),'float_tolerance':1e-6,'cases':checks,'all_221_native_input_acceptance':'SEPARATE_GATE'}
 save(REPORTS/'VECTORIZED_EVIDENCE_TESTS.json',result);print('CANONICAL_EVIDENCE_EQUIVALENCE_PASS',max(x['max_abs_error']for x in checks),flush=True)
if __name__=='__main__':main()
