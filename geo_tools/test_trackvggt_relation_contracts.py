import ast
import unittest
import torch
from common import ROOT
from gtr.geometry.identity_relation_student import TinyRelationStudent
from gtr.geometry.identity_relation_adapter import IdentityRelationAdapter
from gtr.geometry.temporal_identity_cycle import temporal_cycle_loss


class Tests(unittest.TestCase):
    def test_off_zero_and_empty_are_same_raw_object(self):
        logits=torch.randn(2,3);tracks=torch.tensor([4,4,8]);candidate=torch.tensor([[4,8],[8,4]])
        for adapter,delta in [(IdentityRelationAdapter(False),torch.ones(2,2)),(IdentityRelationAdapter(),torch.zeros(2,2)),(IdentityRelationAdapter(max_residual=0),torch.ones(2,2))]:
            self.assertIs(adapter(logits,tracks,candidate,delta),logits)
        for n,m in [(0,3),(2,0),(0,0)]:
            value=torch.empty(n,m)
            self.assertIs(IdentityRelationAdapter()(value,torch.empty(m,dtype=torch.long),torch.empty(n,0,dtype=torch.long),torch.empty(n,0)),value)

    def test_one_track_many_proposals_not_repeated_evidence(self):
        logits=torch.zeros(1,4);tracks=torch.tensor([9,3,9,9]);candidates=torch.tensor([[9,3]])
        result=IdentityRelationAdapter()(logits,tracks,candidates,torch.tensor([[.9,-.2]]))
        self.assertTrue(torch.allclose(result,torch.tensor([[.3,-.2,.3,.3]])))
        self.assertAlmostEqual(float(result[0,tracks==9].sum()),.9,places=6)
        permutation=torch.tensor([2,3,1,0])
        permuted=IdentityRelationAdapter()(logits[:,permutation],tracks[permutation],candidates,torch.tensor([[.9,-.2]]))
        self.assertTrue(torch.equal(result[:,permutation],permuted))

    def test_unknown_unmatched_and_nonfinite_evidence_are_neutral(self):
        logits=torch.randn(1,2);tracks=torch.tensor([3,3])
        for candidate,delta in [(torch.tensor([[-1]]),torch.tensor([[1.]])),(torch.tensor([[3]]),torch.tensor([[float('nan')]])),(torch.tensor([[3]]),torch.tensor([[float('inf')]]))]:
            self.assertIs(IdentityRelationAdapter()(logits,tracks,candidate,delta),logits)
        with self.assertRaises(ValueError):IdentityRelationAdapter()(logits,tracks,torch.tensor([[3,3]]),torch.ones(1,2))

    def test_legal_candidate_mask_preserves_same_camera_exclusion(self):
        logits=torch.zeros(2,2);tracks=torch.tensor([7,42]);candidate=torch.tensor([[7,42],[7,42]])
        legal=torch.tensor([[True,False],[False,True]])
        result=IdentityRelationAdapter()(logits,tracks,candidate,torch.ones(2,2),legal)
        self.assertTrue(torch.equal(result,torch.eye(2)))

    def test_student_zero_init_and_no_teacher_dependency(self):
        model=TinyRelationStudent(23)
        self.assertEqual(float(model(torch.randn(3,23)).abs().sum()),0)
        tree=ast.parse((ROOT/'gtr/geometry/identity_relation_student.py').read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertEqual(imports,['torch'])

    def test_cycle_null_visibility_and_nonzero_timestamp(self):
        ab=torch.tensor([[.8,.2]],requires_grad=True);ba=torch.tensor([[.7,.3]],requires_grad=True);aa=torch.tensor([[.9,.1]],requires_grad=True)
        loss=temporal_cycle_loss(ab,ba,aa,torch.tensor([True]),1001,1002)
        loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(float(ab.grad.abs().sum()),0)
        self.assertEqual(float(temporal_cycle_loss(ab,ba,aa,torch.tensor([False]),1001,1002)),0)
        with self.assertRaises(ValueError):temporal_cycle_loss(ab,ba,aa,torch.tensor([True]),1002,1001)

    def test_invisible_bridge_has_no_fabricated_cycle(self):
        ab=torch.ones(2,1);ba=torch.empty(0,3);aa=torch.tensor([[0.,0.,1.],[.3,.3,.4]])
        self.assertEqual(float(temporal_cycle_loss(ab,ba,aa,torch.tensor([False,False]),6,7)),0)
        bad=aa.clone();bad[0,0]=float('nan')
        with self.assertRaises(ValueError):temporal_cycle_loss(ab,ba,bad,torch.tensor([False,False]),6,7)


if __name__=='__main__':unittest.main()
