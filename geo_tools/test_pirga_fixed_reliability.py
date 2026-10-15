import unittest
import torch
import torch.nn.functional as F
from dataclasses import replace
from cv_pil_v2_model import ActorBatch,fused_scores
from pirga_fixed_reliability import prototypes,METHODS


class FixedReliabilityTests(unittest.TestCase):
    def test_same_full_1152_histories_no_query_in_weights_and_empty_columns_preserved(self):
        torch.manual_seed(3);h=F.normalize(torch.randn(5,1152),dim=-1)
        b=ActorBatch(F.normalize(torch.randn(1152),dim=0),h,torch.tensor([0,0,0,1,1]),torch.ones(5),torch.ones(5,dtype=torch.bool),torch.tensor([.2,.3,.4]),1)
        lag=torch.tensor([4.,1.,2.,3.,2.]);p=prototypes(b,lag,16.)
        torch.testing.assert_close(b.native_scores+2*(p['R_mean']@b.query),fused_scores(b,h),rtol=0,atol=0)
        other=prototypes(replace(b,query=torch.zeros(1152)),lag,16.)
        for method in METHODS:
            self.assertEqual(p[method].shape,(3,1152));self.assertTrue((p[method][2]==0).all())
            torch.testing.assert_close(p[method],other[method],rtol=0,atol=0)
        torch.testing.assert_close(p['R_latest'][0],h[1])
        with self.assertRaises(ValueError):prototypes(b,torch.zeros(5),16.)

if __name__=='__main__':unittest.main()
