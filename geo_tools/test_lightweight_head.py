import unittest,torch
from common import ROOT
from gtr.geometry.lightweight_head import LowRankIdentityHead,matched_hidden_width
from gtr.geometry.pair_score_adapter import PairScoreAdapter

class Tests(unittest.TestCase):
    def test_zero_initial_evidence_and_small_target_memory(self):
        head=LowRankIdentityHead(8,12,hidden_dim=16,output_dim=4,symmetric=True)
        q,r=torch.randn(2,8),torch.randn(3,8);qf,rf=torch.randn(2,24),torch.randn(3,24)
        score=head(q,r,qf,rf);base=torch.randn(2,3)
        self.assertTrue(torch.equal(score,torch.zeros_like(score)))
        self.assertTrue(torch.equal(PairScoreAdapter(True,2,0,2).apply(base,score),base))
        self.assertEqual(head.encode(q,qf).shape,(2,8))
        self.assertEqual(head.encode(q[:0],qf[:0]).shape,(0,8))

    def test_shared_projection_order_invariance_and_gradients(self):
        head=LowRankIdentityHead(8,12,hidden_dim=16,output_dim=4,symmetric=True)
        with torch.no_grad():head.scale.fill_(2.);head.bias.fill_(-.1)
        q,r=torch.randn(2,8),torch.randn(3,8);qf,rf=torch.randn(2,24),torch.randn(3,24)
        swap=lambda x:torch.cat([x[:,12:],x[:,:12]],dim=1)
        score=head(q,r,qf,rf)
        self.assertTrue(torch.allclose(score,head(q,r,swap(qf),swap(rf)),atol=1e-6))
        self.assertTrue(torch.allclose(score,head(r,q,rf,qf).T,atol=1e-6))
        score.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_parameter_matching_uses_actual_dimensions(self):
        vg=LowRankIdentityHead(1152,2048,hidden_dim=96,symmetric=True)
        target=sum(p.numel() for p in vg.parameters())
        for extra in [0,256]:
            hidden=matched_hidden_width(1152,extra,target)
            control=LowRankIdentityHead(1152,extra,hidden_dim=hidden)
            count=sum(p.numel() for p in control.parameters())
            self.assertLess(abs(count-target)/target,.005)

if __name__=='__main__':unittest.main()
