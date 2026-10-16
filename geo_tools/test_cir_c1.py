"""C1 mathematical and causal contract, before opening the native integration."""
import unittest
import torch
import torch.nn.functional as F
import common
from gtr.modeling.identity_revision.features import c1_scores


class C1Contracts(unittest.TestCase):
    def sample(self):
        g=torch.Generator().manual_seed(17)
        return (torch.zeros(2,3),torch.randn(2,1152,generator=g),torch.randn(4,1152,generator=g),
                torch.tensor([3,3,9,20]),torch.tensor([3,9,20]),torch.tensor([1,2,3,3]),3)
    def test_exact_registered_formula(self):
        args=self.sample();out=c1_scores(*args)
        h=F.normalize(args[2][:2],dim=-1);proto=F.normalize(h.sum(0),dim=0)
        expected=2*(F.normalize(args[1],dim=-1)@proto)
        torch.testing.assert_close(out[:,0],expected)
        self.assertTrue(torch.equal(out[:,1:],torch.zeros(2,2)))
    def test_same_timestamp_not_predictive_history(self):
        args=list(self.sample());before=c1_scores(*args)
        args[2][2:]=999
        self.assertTrue(torch.equal(before,c1_scores(*args)))
    def test_future_rejected(self):
        args=list(self.sample());args[5][0]=4
        with self.assertRaises(ValueError):c1_scores(*args)
    def test_no_gt_argument(self):
        with self.assertRaises(TypeError):c1_scores(*self.sample(),gt_identity=9)
    def test_native_candidate_legality(self):
        args=list(self.sample());args[3][0]=123
        with self.assertRaises(ValueError):c1_scores(*args)
    def test_1152_contract(self):
        args=list(self.sample());args[1]=args[1][:,:1024]
        with self.assertRaises(ValueError):c1_scores(*args)
    def test_empty_history(self):
        args=list(self.sample());args[2]=torch.empty(0,1152);args[3]=torch.empty(0,dtype=torch.long);args[5]=torch.empty(0,dtype=torch.long)
        self.assertTrue(torch.equal(c1_scores(*args),args[0]))
    def test_empty_query_and_candidates(self):
        out=c1_scores(torch.empty(0,0),torch.empty(0,1152),torch.empty(0,1152),
            torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long),1)
        self.assertEqual(tuple(out.shape),(0,0))
    def test_no_source_vector_mutation(self):
        args=self.sample();before=[a.clone() for a in args[:-1]]
        c1_scores(*args)
        for a,b in zip(args[:-1],before):self.assertTrue(torch.equal(a,b))


if __name__=='__main__':unittest.main()
