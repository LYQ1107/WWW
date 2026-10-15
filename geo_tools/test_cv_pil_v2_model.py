"""Meaningful isolation, alias, loss-mask and matched-control checks."""
import copy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_cv_pil_native_handoff import fixture
from cv_pil_native_handoff import normalize_event,inference_inputs
from cv_pil_v2_model import CandidatePredictor,tensorize,fused_scores,supervised_losses,parameter_digest


class TestMatchedModel(unittest.TestCase):
    def sample(self):
        source=fixture()
        source['payload']['observations'][(100,0)]['feature'][10]=4.
        source['payload']['observations'][(101,0)]['feature'][30]=4.
        return normalize_event(source,0)

    def test_capacity_initialization_and_c4_camera_removal(self):
        models=[]
        for arm in ('C2','C3','C4'):
            torch.manual_seed(20261015);models.append(CandidatePredictor(use_camera=arm!='C4'))
        self.assertEqual({sum(p.numel() for p in m.parameters()) for m in models},{38080})
        self.assertEqual(len({parameter_digest(m) for m in models}),1)
        batch=tensorize(inference_inputs(self.sample()))
        for m in models:self.assertTrue(torch.allclose(m(batch)[0],fused_scores(batch,batch.histories),atol=1e-6))
        # After nonzero learned weights, no-camera output remains invariant to
        # destination camera while actual explicit camera weights can affect C3.
        m=models[2];torch.nn.init.normal_(m.up.weight,std=.1)
        other=copy.copy(batch);object.__setattr__(other,'destination_camera',1)
        self.assertTrue(torch.equal(m(batch)[0],m(other)[0]))

    def test_actor_rejects_labels_and_scores_ignore_offline_gt(self):
        r=self.sample();m=CandidatePredictor();batch=tensorize(inference_inputs(r));before=m(batch)[0]
        r['OFFLINE_ONLY'].update(GT_person=999,certified_target_indices=(0,))
        self.assertTrue(torch.equal(before,m(tensorize(inference_inputs(r)))[0]))
        with self.assertRaises(TypeError):m(r)
        actor=dict(r['INPUT_ONLY']);actor['GT_person']=999
        with self.assertRaises(ValueError):tensorize(actor)

    def test_alias_multitarget_loss_unknown_mask_and_target_stopgrad(self):
        r=self.sample();batch=tensorize(inference_inputs(r));m=CandidatePredictor()
        score,pred=m(batch);supervision=dict(certified_target_indices=(0,1),candidate_known_mask=(True,True))
        rank,aux,_=supervised_losses(score,pred,batch,supervision)
        self.assertAlmostEqual(float(rank),0.,places=6)
        # An UNKNOWN candidate's logit has no effect on supervised ranking loss.
        sup=dict(certified_target_indices=(1,),candidate_known_mask=(False,True))
        a=supervised_losses(score,pred,batch,sup)[0]
        modified=score.clone();modified[0]+=1000
        b=supervised_losses(modified,pred,batch,sup)[0]
        self.assertEqual(float(a),float(b))
        (rank+aux).backward()
        self.assertFalse(batch.query.requires_grad)
        self.assertTrue(any(x.grad is not None and torch.isfinite(x.grad).all() for x in m.parameters()))

    def test_empty_histories_are_legal_zero_correction(self):
        r=self.sample();r['INPUT_ONLY']['past_histories']=[[],[]]
        b=tensorize(inference_inputs(r));m=CandidatePredictor()
        self.assertTrue(torch.equal(m(b)[0],b.native_scores))

    def test_past_predictor_cannot_read_current_query_target(self):
        b=tensorize(inference_inputs(self.sample()));m=CandidatePredictor()
        other=copy.copy(b);object.__setattr__(other,'query',torch.randn(1152))
        self.assertTrue(torch.equal(m.project(b),m.project(other)))


if __name__=='__main__':unittest.main()
