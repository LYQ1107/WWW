"""Unknown and censored targets cannot become negative training gradients."""
import torch
from jev_phase10_learning import *

def main():
 torch.manual_seed(20261008);x=torch.randn(2,4,2,requires_grad=True);known=torch.tensor([[1,1,0,0],[0,0,0,0]],dtype=torch.bool);positive=torch.tensor([[1,0,0,0],[0,0,0,0]],dtype=torch.bool);q=torch.tensor([[3.,1.,float('nan'),float('nan')],[float('nan')]*4]);qm=torch.tensor([[1,1,0,0],[0,0,0,0]],dtype=torch.bool)
 ce,co,rank,ro=losses(x,known,positive,q,qm);loss=(ce+rank).sum();loss.backward();assert torch.equal(x.grad[:,2:,:],torch.zeros_like(x.grad[:,2:,:]));assert torch.equal(x.grad[1],torch.zeros_like(x.grad[1]));assert co.tolist()==[True,False]and ro.tolist()==[True,False];assert torch.isfinite(loss)
 for name in ['CandidateMLP','CandidateDeepSets','CandidateJEV']:
  model=build_candidate_model(name);st=torch.randn(2,64);ev=torch.randn(2,4,12);mask=torch.ones(2,4,dtype=torch.bool);logits=model(st,ev,mask);ce,_,rank,_=losses(logits,known,positive,q,qm);(ce+rank).sum().backward();assert all(p.grad is None or torch.isfinite(p.grad).all()for p in model.parameters())
 save(REPORTS/'UNKNOWN_NAN_SUPERVISION_TESTS.json',{'status':'PASS','unknown_correctness_gradients_zero':True,'unexecuted_utility_gradients_zero':True,'all_unknown_rows_no_label_gradient':True,'three_model_dual_head_gradients_finite':True,'training_or_tiny_research_fit_performed':False});print('MASKED_SUPERVISION_PASS')
if __name__=='__main__':main()
