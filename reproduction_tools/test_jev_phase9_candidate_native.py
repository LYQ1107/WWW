"""Adversarial shared solver and numeric producer contract tests."""
import unittest,json
import numpy as np
import torch
from jev_phase9_common import *
from gtr.modeling.jev_candidate_assignment import assign_candidate_values as solve
from gtr.modeling.jev_candidate_policy import CandidateValuePolicy
from gtr.modeling.jev_candidate_features import CandidateBatch,evidence12

class CandidateContracts(unittest.TestCase):
 def call(self,s,ids,new=None,mask=None,**kw):
  s=np.array(s,dtype=float);return solve(s,np.zeros(len(s))if new is None else new,ids,np.ones(s.shape,dtype=bool)if mask is None else mask,**kw)
 def test_global_conflict_requires_joint_solve(self):
  a=self.call([[10,9],[9,0]],[101,202],[-100,-100]);self.assertEqual(a.existing_ids,(202,101))
 def test_each_NEW_has_private_dummy(self):self.assertEqual(self.call([[1],[1]],[101],[2,2]).existing_ids,(-1,-1))
 def test_cross_camera_shared_ID_legal(self):self.assertEqual(self.call([[9],[9]],[101],views=[0,1]).existing_ids,(101,101))
 def test_within_camera_capacity(self):a=self.call([[9],[8]],[101]);self.assertEqual(sum(t==101 for t in a.existing_ids),1)
 def test_empty_candidates_and_no_active_tracks(self):a=solve(np.empty((3,0)),[0,0,0],[],np.empty((3,0),bool));self.assertEqual(a.new_rows,(0,1,2))
 def test_zero_detections(self):a=solve(np.empty((0,5)),[],range(5),np.empty((0,5),bool));self.assertEqual(a.existing_ids,())
 def test_all_masked(self):self.assertEqual(self.call([[10,20]],[1,2],mask=[[False,False]]).existing_ids,(-1,))
 def test_nonfinite_existing_illegal(self):self.assertEqual(self.call([[np.nan,np.inf,-np.inf,2]],[1,2,3,4]).existing_ids,(4,))
 def test_nonfinite_NEW_rejected(self):
  with self.assertRaises(ValueError):self.call([[1]],[1],[np.nan])
 def test_duplicate_IDs_rejected(self):
  with self.assertRaises(ValueError):self.call([[1,2]],[1,1])
 def test_tied_permutation_equivariant(self):
  a=self.call([[1,1],[1,1]],[20,10]);b=self.call([[1,1],[1,1]],[10,20]);self.assertEqual(a.existing_ids,b.existing_ids)
 def test_mask_and_column_permutation(self):
  s=np.array([[8,9,6],[9,7,8]]);mask=np.array([[1,0,1],[1,1,1]],bool)
  a=self.call(s,[10,20,30],mask=mask);perm=[2,0,1];b=self.call(s[:,perm],[30,10,20],mask=mask[:,perm]);self.assertEqual(a.existing_ids,b.existing_ids)
 def test_high_candidate_count(self):a=self.call(np.arange(8192).reshape(2,4096),range(4096));self.assertEqual(len(set(a.existing_ids)),2)
 def test_GMT_compatibility_rectangular_then_threshold(self):
  from scipy.optimize import linear_sum_assignment
  s=np.array([[10.,9.],[8.,0.]]);thresholds=[9.,0.];expected=[-1,-1]
  r,c=linear_sum_assignment(-s)
  for i,j in zip(r,c):
   if s[i,j]>thresholds[j]:expected[i]=[10,20][j]
  a=self.call(s,[10,20],mode='gmt_compat',legacy_thresholds=thresholds);self.assertEqual(a.existing_ids,tuple(expected));self.assertFalse(a.semantic_new)
 def test_ID_numbers_not_model_features(self):
  s=torch.tensor([[2.,1.]]);pairs={0:0};length=torch.ones(2)
  a=evidence12(s,[10,20],pairs,length,hits={10:3,20:4});b=evidence12(s,[900,800],pairs,length,hits={900:3,800:4});self.assertTrue(torch.equal(a,b))
 def test_model_output_shapes_and_reference_exclusion(self):
  class ValueProducer(torch.nn.Module):
   def forward(self,state,evidence,mask):return evidence[...,0],state[:,0]
  b=CandidateBatch((12,99),torch.tensor([[2.,1.]]),torch.zeros(1,64),torch.zeros(1,2,12),torch.ones(1,2,dtype=torch.bool),torch.zeros(2),0)
  values,new=CandidateValuePolicy('model',ValueProducer()).score(b);self.assertEqual(values.shape,(1,2));self.assertEqual(new.shape,(1,))
 def test_bidirectional_all_masked_finite(self):
  b=CandidateBatch((1,2),torch.ones(1,2),torch.zeros(1,64),torch.zeros(1,2,12),torch.zeros(1,2,dtype=torch.bool),torch.zeros(2),0)
  v,n=CandidateValuePolicy('bidirectional').score(b);self.assertTrue(torch.isfinite(v).all());self.assertTrue(torch.equal(v,torch.zeros_like(v)))
 def test_shape_mismatch(self):
  with self.assertRaises(ValueError):solve([[1]],[0],[1],[[True,False]])
if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(CandidateContracts);r=unittest.TextTestRunner(verbosity=2).run(suite)
 save(REPORTS/'CANDIDATE_INTERFACE_TESTS.json',{'status':'PASS'if r.wasSuccessful()else'FAIL','tests':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'production_full_commit_parity':'PENDING','binding':binding()});sys.exit(0 if r.wasSuccessful()else 1)
