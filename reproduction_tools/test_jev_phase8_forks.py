import unittest
from types import SimpleNamespace
import torch
from jev_phase8_common import REPORTS,save
from jev_phase8_candidate_submit import validate_pairs
from jev_phase8_utility import effects

class ForkContracts(unittest.TestCase):
    def proposal(self,ids=(1,7),banned=()):return SimpleNamespace(track_ids=ids,scores=torch.ones(2,2),banned_edges=banned)
    def test_unique_claims(self):
        with self.assertRaises(AssertionError):validate_pairs(self.proposal(),{0:0,1:0})
    def test_duplicate_candidates(self):
        with self.assertRaises(AssertionError):validate_pairs(self.proposal((7,7)),{0:0})
    def test_mask(self):
        with self.assertRaises(AssertionError):validate_pairs(self.proposal(banned=((0,1),)),{0:1})
    def test_missing(self):
        with self.assertRaises(AssertionError):validate_pairs(self.proposal(),{0:2})
    def test_empty_new(self):self.assertEqual(validate_pairs(self.proposal(),{}),{})
    def prefix(self):return {'reliable':{'1':8,'7':9},'counts':{'1':{'8':10},'7':{'9':10}},'last_GT_frame':{'8':0,'9':0}}
    def test_anchored_wrong_write_never_renames(self):
        e=effects({(1,0,0):{'id':1,'gt':9},(2,0,0):{'id':1,'gt':9}},self.prefix(),1,8,{9})
        self.assertEqual(e['counts']['wrong'],2);self.assertEqual(e['wrong_identity_duration_camera_frames'],2)
        self.assertEqual(e['utility_birth_zero'],-7)
    def test_identity_integer_collision_not_correct(self):
        e=effects({(1,0,0):{'id':7,'gt':7}},self.prefix(),1,8,{7});self.assertEqual(e['counts']['wrong'],1)
    def test_unconfirmed_existing_stays_unknown(self):
        p=self.prefix();p['counts']['3']={'8':1,'9':1}
        e=effects({(1,0,0):{'id':3,'gt':8},(2,0,0):{'id':3,'gt':8}},p,1,8,{8});self.assertEqual(e['counts']['anchor_unknown'],2)
    def test_new_confirms_only_after_two(self):
        e=effects({(1,0,0):{'id':42,'gt':8},(2,0,0):{'id':42,'gt':8}},self.prefix(),1,8,{8})
        self.assertEqual(e['counts']['anchor_unknown'],1);self.assertEqual(e['counts']['correct'],1);self.assertEqual(e['counts']['false_births'],1)
        self.assertEqual(e['utility_birth_zero'],1)
    def test_horizon_no_future(self):
        e=effects({(9,0,0):{'id':1,'gt':9}},self.prefix(),1,8,{9});self.assertEqual(e['utility'],0)
    def test_unknown_identity_censors_instead_of_claiming_recovery(self):
        p=self.prefix();p['counts']['3']={'8':1,'9':1}
        e=effects({(1,0,0):{'id':1,'gt':9},(2,0,0):{'id':3,'gt':9},(3,0,0):{'id':7,'gt':9}},p,1,8,{9})
        self.assertEqual(e['wrong_identity_duration_camera_frames'],1)
        self.assertTrue(e['wrong_identity_episodes'][0]['right_censored'])

if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ForkContracts))
    save(REPORTS/'FORK_CONTRACT_TESTS.json',{'tests':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'status':'PASS'if r.wasSuccessful()else'FAIL',
        'scope':'adversarial pair legality + temporal offline utility; no claim of end-to-end deployed production parity'})
    raise SystemExit(not r.wasSuccessful())
