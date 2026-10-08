"""No research labels: numeric, permutation, mask and scripted-interface checks."""
import torch
from jev_phase10_common import *
from gtr.modeling.jev_candidate_models import build_candidate_model,CandidateScorer

def main():
 torch.manual_seed(20261008);results=[]
 names=['CandidateMLP','CandidateDeepSets','CandidateJEV','CandidateJEV_without_QA','CandidateJEV_without_context']
 for name in names:
  model=build_candidate_model(name).eval();script=torch.jit.script(CandidateScorer(model));maxerror=0.
  for rows,candidates in [(0,0),(3,0),(0,7),(1,1),(5,17),(8,65)]:
   state=torch.randn(rows,64);evidence=torch.randn(rows,candidates,12);mask=torch.rand(rows,candidates)>.2
   if rows and candidates:mask[0]=False
   with torch.no_grad():y=model(state,evidence,mask);values,new=script(state,evidence,mask)
   assert torch.isfinite(y).all()and torch.isfinite(values).all()and torch.isfinite(new).all()
   assert y.shape==(rows,candidates,2)and values.shape==(rows,candidates)and new.shape==(rows,)
   order=torch.randperm(candidates)
   with torch.no_grad():permuted=model(state,evidence[:,order],mask[:,order]);corrupt=evidence.clone();corrupt[~mask]=10000.;altered=model(state,corrupt,mask)
   err=float((permuted-y[:,order]).abs().max())if y.numel()else 0.;maxerror=max(maxerror,err)
   assert err<=1e-6 and torch.equal(altered,y),'permutation/masked evidence affected legal scores'
  params=sum(p.numel()for p in model.parameters());results.append({'model':name,'parameters':params,'trainable_parameters':sum(p.numel()for p in model.parameters()if p.requires_grad),'max_permutation_abs_error':maxerror,'scripted_finite_empty_all_masked_and_shared_interface':'PASS'})
 pcounts=[r['parameters']for r in results[:3]];assert max(pcounts)/min(pcounts)<1.05
 save(REPORTS/'CANDIDATE_MODEL_CAPACITY_ABLATION_TESTS.json',{'status':'PASS','models':results,'research_training_performed':False,'tiny_overfit_not_yet_run':True,'heldout_sealed':True});print(results)
if __name__=='__main__':main()
