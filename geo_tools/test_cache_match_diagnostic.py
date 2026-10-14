import copy
import unittest
from types import SimpleNamespace
import torch
from common import ROOT
from gtr.geometry.target_feature_cache import TargetFeatureCache
from gtr.geometry.cache_match_diagnostic import match_gates


class Tests(unittest.TestCase):
    def test_recomputes_actual_rejection_without_mutation_and_exposes_spatial_pool(self):
        cache=TargetFeatureCache()
        cache.store(1,0,torch.tensor([[1.,0.],[.99,.01]]),torch.tensor([[0.,0.,10.,10.],[50.,50.,60.,60.]]),
                    torch.ones(2,1),torch.ones(2,dtype=torch.bool),torch.tensor([.9,.8]))
        native=torch.tensor([[1.,.001]]);boxes=torch.tensor([[0.,0.,10.,10.]])
        _,valid,_=cache.reuse(1,6,native,boxes,1)
        self.assertEqual(valid.tolist(),[False])
        query=SimpleNamespace(reid_features=native,geo_xyxy=boxes,geo_camera=torch.tensor([1]),geo_frame=torch.tensor([6]),geo_valid=valid)
        stats=copy.deepcopy(cache.stats);rng=torch.get_rng_state().clone()
        result=match_gates(cache,query)
        self.assertTrue(result['original_matches_actual_valid'])
        self.assertEqual(result['original']['gate_pass']['relative_margin'],[False])
        self.assertEqual(result['spatially_admissible_counts'],[1])
        self.assertEqual(result['spatial_filter_counterfactual']['accepted'],[True])
        self.assertEqual(cache.stats,stats)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_fresh_snapshot_is_not_falsely_judged_as_a_reuse_event(self):
        cache=TargetFeatureCache()
        native=torch.tensor([[1.,0.]]);boxes=torch.tensor([[0.,0.,1.,1.]])
        cache.store(1,0,native,boxes,torch.ones(1,1),torch.tensor([True]),torch.tensor([1.]))
        query=SimpleNamespace(reid_features=native,geo_xyxy=boxes,geo_camera=torch.tensor([1]),geo_frame=torch.tensor([0]),geo_valid=torch.tensor([True]))
        self.assertEqual(match_gates(cache,query)['status'],'NOT_A_CURRENT_REUSE_EVENT')


if __name__=='__main__':unittest.main()
