import unittest,itertools
import numpy as np
import torch
import torch.nn.functional as F
from pirga_native_context import opportunity
from pirga_candidate_state import CandidateState
from pirga_decision_head import DecisionMLP,GroupedDecisions,decision_input,decision_losses


class NativeDecisionTests(unittest.TestCase):
    def test_full_assignment_opportunity_matches_exhaustive_forced_row_cost(self):
        a=np.array([[.9,.8,.1],[.85,.1,.4],[.1,.7,.6]],np.float32)
        costs=opportunity(a,0).numpy()
        others=a[1:];best=max(sum(others[r,c] for r,c in enumerate(cols)) for cols in itertools.permutations(range(3),2))
        expected=[max(sum(others[r,c] for r,c in enumerate(cols)) for cols in itertools.permutations([i for i in range(3) if i!=j],2))-best for j in range(3)]
        np.testing.assert_allclose(costs,expected,atol=1e-7)
        np.testing.assert_array_equal(opportunity(np.empty((2,0)),1),[])

    def test_exact_c1_initialization_and_independent_optimizer_math(self):
        torch.set_num_threads(1);torch.manual_seed(3)
        st=CandidateState(F.normalize(torch.randn(3,1152),dim=-1),torch.randn(3,11),torch.tensor([4.,3.,0.]),torch.tensor([1.,2.,0.]),torch.tensor([3.,2.,0.]))
        q=F.normalize(torch.randn(1152),dim=0);native=torch.tensor([.4,.2,.1]);threshold=torch.tensor([.2,.3,.1]);opp=torch.tensor([-.2,-.1,0.]);x=decision_input(st,q,native,threshold,opp,0)
        group=GroupedDecisions(8);models=[DecisionMLP(),DecisionMLP()]
        for i,m in enumerate(models):m.load_state_dict(group.arm_state(i))
        torch.testing.assert_close(group(x,st,q,native)[0][0],native+2*(st.mean@q),rtol=0,atol=0)
        label=dict(GT_person=7,certified_target_indices=[0],candidate_known_mask=[True,True,False],actual_primary_state='correct_link')
        opt=torch.optim.AdamW(group.parameters(),lr=.001,weight_decay=.0001);opts=[torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001) for m in models]
        for _ in range(5):
            sc,de=group(x,st,q,native);loss=torch.stack([decision_losses(sc[i],de[i],label,threshold,opp,0,bool(i))[0] for i in range(2)])
            opt.zero_grad(set_to_none=True);loss.sum().backward();group.clip_per_arm();opt.step()
            for i,m in enumerate(models):
                s,d=m(x,st,q,native);l=decision_losses(s,d,label,threshold,opp,0,bool(i))[0]
                torch.testing.assert_close(s,sc[i],atol=1e-5,rtol=1e-5);opts[i].zero_grad(set_to_none=True);l.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opts[i].step()
                for name,value in m.state_dict().items():torch.testing.assert_close(value,group.arm_state(i)[name],atol=5e-4,rtol=0)
        with self.assertRaises(ValueError):decision_losses(sc[0],de[0],dict(label,GT_person=None),threshold,opp,0,True)
        altered=sc[1].detach().clone();altered[2]=1e6
        a=decision_losses(sc[1],de[1],label,threshold,opp,0,True)[0];b=decision_losses(altered,de[1],label,threshold,opp,0,True)[0]
        torch.testing.assert_close(a,b,rtol=0,atol=0)

if __name__=='__main__':unittest.main()
