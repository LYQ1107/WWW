import unittest
import torch
from common import ROOT
from trackvggt_data import association_losses
from evaluate_teacher_pairs import predicted_state
from gtr.geometry.track_ag_teacher import TrackRelationTeacher


class Tests(unittest.TestCase):
    def features(self,zero=False):
        return dict(histories=[dict(features=torch.randn(3,11),times=torch.tensor([1001.,1002.,1003.]),cameras=torch.tensor([7,42,7]),observed_length=5) for _ in range(2)],
            pair=torch.randn(2,8),time=1003,camera=42,native_scores=torch.tensor([1.2,.8]),current=torch.randn(11),
            layers=[torch.randn(7,8) for _ in range(4)],roi=torch.randn(4,8),attention=torch.rand(7),
            xy=torch.randn(7,2),inside=torch.zeros(7,dtype=torch.bool) if zero else torch.tensor([True]*3+[False]*4))

    def test_ag_only_modes_and_no_qd_shortcut(self):
        for mode in ('roi','A','B','C'):
            model=TrackRelationTeacher(11,8,architecture='ag',ag_mode=mode)
            self.assertIsNone(model.qd)
            score,aux=model(self.features(),soft=mode=='C')
            self.assertEqual(score.shape,(3,));self.assertTrue(torch.isfinite(score).all());self.assertFalse(aux['neutral'])

    def test_c_hard_stage_still_trains_relevance(self):
        model=TrackRelationTeacher(11,8,ag_mode='C');score,_=model(self.features(),soft=False)
        target=dict(positive=torch.tensor([True,False,False]),known=torch.ones(3,dtype=torch.bool),loss_eligible=True)
        loss=association_losses(score,target)[0];loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(float(model.sampler.mix.grad.abs().sum()),0)

    def test_zero_patch_stays_in_evaluation_as_native_fallback(self):
        model=TrackRelationTeacher(11,8);features=self.features(True);score,aux=model(features)
        self.assertTrue(aux['neutral']);self.assertTrue(torch.equal(score[:2],features['native_scores']))
        self.assertEqual(float(score[-1]),0)

    def test_empty_gallery_returns_only_explicit_null(self):
        model=TrackRelationTeacher(11,8);features=self.features();features['histories']=[];features['native_scores']=torch.empty(0)
        score,aux=model(features);self.assertEqual(score.tolist(),[0.]);self.assertTrue(aux['neutral'])

    def test_multiple_positive_tracks_and_unknown_not_negative(self):
        logits=torch.tensor([.3,.7,9.,-.2],requires_grad=True)
        target=dict(positive=torch.tensor([True,True,False,False]),known=torch.tensor([True,True,False,True]),loss_eligible=True)
        value=association_losses(logits,target)[0];value.backward()
        self.assertEqual(float(logits.grad[2]),0)
        self.assertLess(float(logits.grad[0]),0);self.assertLess(float(logits.grad[1]),0)
        target['positive'][2]=True
        with self.assertRaises(ValueError):association_losses(logits,target)

    def test_unmatched_and_ambiguous_prediction_states(self):
        target=dict(query_gt_known=True,known_unmatched=False,top8_positive=True,positive=torch.tensor([True,False,False]),ambiguous_indices=[1])
        self.assertEqual(predicted_state([0,1,2],target),'correct_link')
        self.assertEqual(predicted_state([1,0,2],target),'ambiguous_reference')
        self.assertEqual(predicted_state([2,0,1],target),'missed_known_positive')
        target['known_unmatched']=True
        self.assertEqual(predicted_state([2,0,1],target),'correct_null_within_recorded_history')


if __name__=='__main__':unittest.main()
