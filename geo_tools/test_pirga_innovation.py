"""Mechanism controls: query exclusion, exact C1 fallback and UNKNOWN masks."""
import unittest
import torch
import torch.nn.functional as F
from pirga_candidate_state import CandidateState
from pirga_innovation import InnovationHead,innovation_losses,ARMS


class InnovationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8)
        self.state=CandidateState(F.normalize(torch.randn(3,1152),dim=-1),torch.randn(3,11),torch.tensor([5.,2.,0.]),torch.tensor([2.,0.,0.]),torch.tensor([5.,8.,0.]))
        self.query=F.normalize(torch.randn(1152),dim=0);self.native=torch.tensor([.2,.1,.3])
        self.label=dict(GT_person=7,certified_target_indices=[0],candidate_known_mask=[True,True,False])

    def test_zero_innovation_is_exact_c1_and_no_query_enters_predictor(self):
        for arm in ARMS:
            model=InnovationHead(arm);scores,mu,lv=model(self.state,self.query,self.native)
            torch.testing.assert_close(scores,self.native+2*(self.state.mean@self.query),rtol=0,atol=0)
            other_query=F.normalize(torch.randn(1152),dim=0)
            other=model(self.state,other_query,self.native)
            torch.testing.assert_close(mu,other[1],rtol=0,atol=0)
            torch.testing.assert_close(lv,other[2],rtol=0,atol=0)
            model.up.bias.data.normal_()
            fallback=model(self.state,self.query,self.native,no_innovation=True)[0]
            torch.testing.assert_close(fallback,self.native+2*(self.state.mean@self.query),rtol=0,atol=0)

    def test_increment_bounded_and_uncertainty_clipped(self):
        model=InnovationHead('A4_uncertainty');model.up.weight.data.normal_();model.sigma.weight.data.normal_(std=1000)
        scores,mu,lv=model(self.state,self.query,self.native)
        self.assertTrue((abs(scores-(self.native+2*self.state.mean@self.query))<=.500001).all())
        self.assertTrue((lv.exp()>=.04999).all() and (lv.exp()<=4.00001).all())

    def test_unknown_candidate_excluded_and_unknown_query_refused(self):
        model=InnovationHead('A4_uncertainty');scores,mu,lv=model(self.state,self.query,self.native)
        loss=innovation_losses(scores,mu,lv,self.state,self.query,self.label,model.arm)[0]
        changed=scores.clone();changed[2]=1e6
        other=innovation_losses(changed,mu,lv,self.state,self.query,self.label,model.arm)[0]
        torch.testing.assert_close(loss,other,rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError,'supervision'):
            innovation_losses(scores,mu,lv,self.state,self.query,dict(self.label,GT_person=None),model.arm)

    def test_camera_and_time_ablations_remove_only_explicit_channels(self):
        for arm,columns in [('A6_no_camera',[0,1,2]),('A7_no_time',[3,4,7,8,9,10])]:
            model=InnovationHead(arm);model.up.weight.data.normal_();mu,lv=model.predict(self.state)
            cond=self.state.conditions.clone();cond[:,columns]=1000
            other=CandidateState(self.state.mean,cond,self.state.counts,self.state.cross_counts,self.state.latest_lag)
            m2,l2=model.predict(other)
            torch.testing.assert_close(mu,m2,rtol=0,atol=0);torch.testing.assert_close(lv,l2,rtol=0,atol=0)

if __name__=='__main__':unittest.main()
