import unittest
import torch
from common import ROOT
from gtr.geometry.target_feature_cache import TargetFeatureCache


class Tests(unittest.TestCase):
    def test_bbox_native_mutual_and_age_gates(self):
        cache=TargetFeatureCache(max_age=2)
        native=torch.eye(2);boxes=torch.tensor([[0.,0,10,10],[20,0,30,10]]);projected=torch.tensor([[1.,0,0],[0,1,0]])
        cache.store(1,0,native,boxes,projected,torch.ones(2,dtype=torch.bool),torch.ones(2))
        values,valid,source=cache.reuse(1,1,native,boxes,3)
        self.assertTrue(valid.all());self.assertTrue(torch.equal(values,projected));self.assertEqual(source.tolist(),[0,0])
        self.assertFalse(cache.reuse(2,1,native,boxes,3)[1].any())
        self.assertFalse(cache.reuse(1,1,native,boxes+100,3)[1].any())
        self.assertEqual(int(cache.reuse(1,1,torch.ones(2,2),boxes,3)[1].sum()),0)
        duplicate=torch.cat([native[:1],native[:1]])
        self.assertLessEqual(int(cache.reuse(1,1,duplicate,boxes[:1].expand(2,-1),3)[1].sum()),1)
        self.assertFalse(cache.reuse(1,3,native,boxes,3)[1].any())

    def test_bounds_detached_snapshot_future_and_nonfinite(self):
        cache=TargetFeatureCache(max_targets=1)
        native=torch.eye(2,requires_grad=True);boxes=torch.tensor([[0.,0,10,10],[20,0,30,10]]);projected=torch.eye(2)
        cache.store(1,5,native,boxes,projected,torch.ones(2,dtype=torch.bool),torch.tensor([.2,.9]))
        self.assertEqual(len(cache.entries[1]['native']),1)
        self.assertFalse(cache.entries[1]['native'].requires_grad)
        self.assertEqual(cache.payload_bytes(),32)
        cache.store(2,5,native,boxes,projected,torch.ones(2,dtype=torch.bool),torch.ones(2))
        with self.assertRaises(ValueError):cache.store(3,5,native,boxes,projected,torch.ones(2,dtype=torch.bool),torch.ones(2))
        with self.assertRaises(ValueError):cache.store(1,4,native,boxes,projected,torch.ones(2,dtype=torch.bool),torch.ones(2))
        with self.assertRaises(ValueError):cache.reuse(1,4,native,boxes,2)
        values,valid,source=cache.reuse(1,6,torch.full((2,2),float('nan')),boxes,2)
        self.assertFalse(valid.any());self.assertTrue(torch.isfinite(values).all());self.assertEqual(source.tolist(),[-1,-1])

    def test_causal_prefix_without_future_camera_updates(self):
        def run(length):
            cache=TargetFeatureCache(max_age=3);result=[];native=torch.eye(2);boxes=torch.tensor([[0.,0,10,10],[20,0,30,10]])
            for frame in range(length):
                if frame%3==0:cache.store(1,frame,native,boxes,torch.ones(2,4)*(frame+1),torch.ones(2,dtype=torch.bool),torch.ones(2))
                result.append(tuple(value.tolist() for value in cache.reuse(1,frame,native,boxes,4)))
            return result
        self.assertEqual(run(8),run(12)[:8])


if __name__=='__main__':unittest.main()
