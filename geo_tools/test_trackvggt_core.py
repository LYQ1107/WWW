import copy
import unittest
import torch
from common import ROOT
from gtr.geometry.roi_feature_extractor import PadMapping
from gtr.geometry.trackvggt_features import explicit_layer_map,incoming_attention
from gtr.geometry.track_query_encoder import TrackQueryEncoder
from gtr.geometry.identity_guided_sampler import greedy_2d,patch_regions,normalize_attention,IdentityGuidedSampler
from gtr.geometry.track_conditioned_see_query import TrackConditionedSeeQuery
from gtr.geometry.track_qd_teacher import TrackQDTeacher


class Tests(unittest.TestCase):
    def test_layer_index_is_not_compact_index(self):
        layers=[None]*24
        for index in (4,11,17,23):layers[index]=torch.full((1,2,9,8),float(index))
        mapped=explicit_layer_map(layers,5,(28,28))
        self.assertEqual(list(mapped),[4,11,17,23])
        self.assertEqual(mapped[17].shape,(1,2,4,8))
        compact=[layers[i] for i in (4,11,17,23)]
        with self.assertRaises(ValueError):explicit_layer_map(compact,5,(28,28))
        self.assertTrue(torch.equal(explicit_layer_map(compact,5,(28,28),(4,11,17,23))[17],mapped[17]))

    def test_attention_equation_chunk_and_orientation(self):
        from types import SimpleNamespace
        torch.manual_seed(4)
        module=SimpleNamespace(qkv=torch.nn.Linear(16,48),num_heads=4,head_dim=4,q_norm=torch.nn.Identity(),k_norm=torch.nn.Identity(),rope=None,scale=.5)
        x=torch.randn(2,11,16)
        q,k,_=module.qkv(x).reshape(2,11,3,4,4).permute(2,0,3,1,4).unbind(0)
        expected=(q@k.transpose(-2,-1)*.5).softmax(-1).mean((1,2))
        self.assertTrue(torch.allclose(incoming_attention(module,x,chunk=3),expected,atol=1e-7))
        self.assertTrue(torch.allclose(expected.sum(-1),torch.ones(2)))

    def test_pad_and_original_xy_orientation(self):
        mapping=PadMapping(1920,1080)
        xy,inside,context,valid=patch_regions(mapping,[0,0,1920,1080])
        self.assertTrue(torch.equal(inside,valid))
        self.assertTrue((xy[valid]>=0).all())
        self.assertTrue((xy[valid][:,0]<1920).all())
        self.assertTrue((xy[valid][:,1]<1080).all())
        self.assertFalse(bool((inside&context).any()))
        _,tiny,_,_=patch_regions(mapping,[1,1,2,2])
        self.assertEqual(int(tiny.sum()),0)

    def test_greedy_coverage_caps_constant_attention(self):
        xy=torch.stack((torch.arange(40),torch.zeros(40)),-1).float()
        inside=torch.arange(40)<20;context=~inside
        score=normalize_attention(torch.ones(40))
        result=greedy_2d(score,xy,inside,context,24)
        self.assertTrue(torch.equal(result,greedy_2d(score,xy,inside,context,24)))
        self.assertEqual(len(result.unique()),24)
        self.assertGreaterEqual(int(inside[result].sum()),12)
        self.assertLessEqual(int(context[result].sum()),12)
        self.assertEqual(int(result[0]),0)
        self.assertEqual(int(normalize_attention(torch.zeros(3)).sum()),0)

    def test_all_padding_and_ag_gradients(self):
        model=IdentityGuidedSampler(12);tokens=torch.randn(30,12);query=torch.randn(128,requires_grad=True)
        mask=torch.zeros(30,dtype=torch.bool);xy=torch.randn(30,2);attention=torch.rand(30)
        selected,indices,_=model(tokens,attention,query,xy,mask,mask)
        self.assertEqual(len(selected),0);self.assertEqual(len(indices),0)
        mask[:10]=True
        pooled,_,_=model(tokens,attention,query,xy,mask,~mask,soft=True)
        pooled.square().sum().backward()
        self.assertGreater(float(model.query.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.key.weight.grad.abs().sum()),0)
        self.assertGreater(float(query.grad.abs().sum()),0)

    def test_history_no_padding_future_or_id_ordinal(self):
        for summary in ('mean','ema'):
            model=TrackQueryEncoder(19,summary=summary)
            history=torch.randn(3,19);times=torch.tensor([10.,11.,12.]);cameras=torch.tensor([7,42,7])
            first=model(history,times,cameras,12,42)
            second=model(history,times,torch.tensor([99,100,99]),12,100)
            self.assertTrue(torch.equal(first,second))
            self.assertEqual(first.shape,(128,))
            with self.assertRaises(ValueError):model(history,times,cameras,11,42)
            with self.assertRaises(ValueError):model(history[:0],times[:0],cameras[:0],12,42)
            with self.assertRaises(ValueError):model(torch.randn(3,20),times,cameras,12,42)

    def test_conditioned_and_global_query_shuffle(self):
        query=torch.randn(2,128)
        dynamic=TrackConditionedSeeQuery().eval();global_query=TrackConditionedSeeQuery(conditioned=False).eval()
        weights=dynamic(query)
        self.assertEqual(weights.shape,(2,4));self.assertTrue((weights>=0).all())
        self.assertTrue(torch.allclose(weights.sum(-1),torch.ones(2)))
        self.assertFalse(torch.allclose(weights[0],weights[1]))
        self.assertTrue(torch.equal(global_query(query)[0],global_query(query)[1]))

    def test_qd_normal_training_tensors_receive_gradient(self):
        with torch.inference_mode():frozen=[torch.randn(5,32) for _ in range(4)]
        layers=[value.clone() for value in frozen]
        model=TrackQDTeacher([32]*4).train()
        score,weights=model(torch.randn(128),layers,torch.randn(8))
        (score.square()+weights[0]).backward()
        for projection in model.projections:self.assertGreater(float(projection[0].weight.grad.abs().sum()),0)
        self.assertGreater(float(model.see.condition[0].weight.grad.abs().sum()),0)
        self.assertGreater(float(model.decoder.layers[0].multihead_attn.in_proj_weight.grad.abs().sum()),0)
        with self.assertRaises(ValueError):model(torch.randn(128),layers,torch.randn(8),torch.zeros(5,dtype=torch.bool))
        layers[0][0,0]=float('nan')
        with self.assertRaises(ValueError):model(torch.randn(128),layers,torch.randn(8))

    def test_qd_ablation_modes_have_finite_scores(self):
        for mode in ('last','mean','concat','global','track'):
            model=TrackQDTeacher([32]*4,mode=mode).eval()
            score,weights=model(torch.randn(128),[torch.randn(3,32) for _ in range(4)],torch.randn(8))
            self.assertTrue(torch.isfinite(score))
            if weights is not None:self.assertAlmostEqual(float(weights.sum()),1.,places=6)

    def test_active_capacity_of_primary_dino_global_track_controls(self):
        def total(channels,bottleneck=None,mode='track'):
            modules=[TrackQueryEncoder(1152),IdentityGuidedSampler(channels,projection_hidden=bottleneck),TrackQDTeacher([channels]*4,projection_hidden=bottleneck,mode=mode)]
            return sum(parameter.numel() for module in modules for parameter in module.parameters())
        self.assertEqual(total(2048),2123016)
        self.assertEqual(total(1024,228),2126488)
        self.assertLess(abs(total(2048)-total(1024,228))/total(2048),.002)
        self.assertEqual(total(2048,mode='global'),total(2048))


if __name__=='__main__':unittest.main()
