import unittest,torch
from common import ROOT
from gtr.geometry.pair_score_adapter import normalize_candidate_scores

class Test(unittest.TestCase):
    def test_bias_invariance(self):
        values=torch.tensor([[10.,12.,999.],[1.,1.,float('nan')]])
        mask=torch.tensor([[True,True,False],[True,True,True]])
        z,valid=normalize_candidate_scores(values,mask)
        self.assertTrue(torch.allclose(z[0],torch.tensor([-1.,1.,0.])))
        self.assertTrue(torch.equal(z[1],torch.zeros(3)));self.assertFalse(valid[1].any())
        z2,_=normalize_candidate_scores(values+42,mask);self.assertTrue(torch.equal(z,z2))
    def test_empty_singleton(self):
        z,valid=normalize_candidate_scores(torch.randn(2,4),torch.zeros(2,4,dtype=torch.bool));self.assertFalse(valid.any());self.assertTrue(torch.equal(z,torch.zeros_like(z)))
        z,valid=normalize_candidate_scores(torch.tensor([[1.]]),torch.tensor([[True]]));self.assertFalse(valid.any());self.assertEqual(float(z.sum()),0.)
if __name__=='__main__':unittest.main()
