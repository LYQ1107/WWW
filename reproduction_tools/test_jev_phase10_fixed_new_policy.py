"""Verify same birth confidence, lawful capacity and neural runtime interface."""
import torch,unittest
from jev_phase10_common import *
from gtr.modeling.jev_candidate_features import CandidateBatch
from gtr.modeling.jev_candidate_models import build_candidate_model,CandidateScorer
from gtr.modeling.jev_candidate_policy import CandidateValuePolicy
from gtr.modeling.jev_candidate_assignment import assign_candidate_values

def main():
 torch.manual_seed(20261008);checked=0;maxerror=0.
 for rows,candidates in [(0,0),(3,0),(0,7),(1,1),(5,17),(8,65)]:
  s=torch.randn(rows,candidates);st=torch.randn(rows,64);e=torch.randn(rows,candidates,12);mask=torch.rand(rows,candidates)>.2
  if rows and candidates:mask[0]=False
  threshold=torch.full((candidates,),.5);batch=CandidateBatch(tuple(range(candidates)),s,st,e,mask,threshold,0)
  policies=[CandidateValuePolicy('gmt_values'),CandidateValuePolicy('dynamic_fixed_new',dynamic_alpha=.2),CandidateValuePolicy('bidirectional_fixed_new')]
  policies += [CandidateValuePolicy('model_fixed_new',torch.jit.script(CandidateScorer(build_candidate_model(name).eval())))for name in ['CandidateMLP','CandidateDeepSets','CandidateJEV']]
  for p in policies:
   v,n=p.score(batch);assert n.shape==(rows,)and torch.equal(n,torch.zeros_like(n));assert torch.isfinite(v[mask]).all()
   if candidates and rows:
    eligible=mask.any(1);actual=v.masked_fill(~mask,-torch.inf).max(1).values;want=(s-threshold).masked_fill(~mask,-torch.inf).max(1).values
    err=float((actual[eligible]-want[eligible]).abs().max())if eligible.any()else 0.;assert err<=1e-6;maxerror=max(maxerror,err)
   a=assign_candidate_values(v,n,batch.candidate_ids,mask);assert len(a.existing_ids)==rows
   order=torch.randperm(candidates);permuted=CandidateBatch(tuple(batch.candidate_ids[i]for i in order),s[:,order],st,e[:,order],mask[:,order],threshold[order],0);vp,np=p.score(permuted);b=assign_candidate_values(vp,np,permuted.candidate_ids,permuted.legal_mask);assert a.existing_ids==b.existing_ids
   checked+=1
 # Run existing tests without their main's write to frozen Phase IX reports.
 import test_jev_phase9_candidate_native as legacy
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(legacy.CandidateContracts);r=unittest.TextTestRunner(verbosity=0).run(suite);assert r.wasSuccessful()
 save(REPORTS/'FIXED_NEW_POLICY_CONTRACT_TESTS.json',{'status':'PASS','new_policy_cases':checked,'legacy_interface_cases':r.testsRun,'max_birth_confidence_abs_error':maxerror,'shared_private_NEW_value':0.,'permutation_and_capacity_exact':True,'research_training':'NOT_RUN','heldout_sealed':True});print('COMMON_FIXED_NEW_PASS',checked,r.testsRun,maxerror)
if __name__=='__main__':main()
