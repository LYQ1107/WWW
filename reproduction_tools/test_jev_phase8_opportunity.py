"""Identity truth and assignment tests that exclude common false opportunities."""
import unittest,json
import numpy as np
from jev_phase8_opportunity import PrefixIdentityAnchors,classify_row,forced_feasible_pairs
from jev_phase8_common import REPORTS,save

class Contracts(unittest.TestCase):
    def anchors(self):
        a=PrefixIdentityAnchors()
        for f in (0,1):a.update({0:17,1:2},{0:2,1:17},f)
        return a
    def test_integer_identity_collision_is_not_correct(self):
        d=classify_row([17,2],[[.9,.8]],{0:0},0,17,'ACCEPT_CURRENT',self.anchors())
        self.assertFalse(d['proposal_correct']);self.assertEqual(d['correct_candidate_ids'],[2])
    def test_aliases_retained(self):
        a=self.anchors();a.update({0:18},{0:17},0);a.update({0:18},{0:17},1)
        d=classify_row([17,2,18],[[.9,.8,.7]],{0:0},0,17,'ACCEPT_CURRENT',a)
        self.assertEqual(d['correct_candidate_ids'],[2,18]);self.assertTrue(d['duplicate_GT_alias_ambiguity'])
    def test_mixed_history_is_unknown(self):
        a=self.anchors();a.update({0:17},{0:17},2)
        self.assertNotIn(17,a.reliable());self.assertIn(17,a.majority())
    def test_first_observation_is_not_stable_anchor(self):
        a=PrefixIdentityAnchors();a.update({0:9},{0:1},0);self.assertEqual(a.reliable(),{})
    def test_forced_global_choice_resolves_conflicting_row(self):
        p=forced_feasible_pairs([[.9,.8],[.8,.9]],[],0,1)
        self.assertEqual(p,{0:1,1:0})
    def test_illegal_edge_not_available(self):
        self.assertIsNone(forced_feasible_pairs([[.9,.8]],[(0,1)],0,1))
    def test_unknown_target_not_a_negative_label(self):
        d=classify_row([17,2],[[.9,.8]],{0:0},0,None,'ACCEPT_CURRENT',self.anchors())
        self.assertFalse(d['strict_wrong_with_feasible_correct_candidate']);self.assertEqual(d['bucket'],'GT_UNKNOWN')

if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    save(REPORTS/'OPPORTUNITY_CONTRACT_TESTS.json',{'status':'PASS'if r.wasSuccessful()else'FAIL','tests':r.testsRun,'failures':len(r.failures),'errors':len(r.errors)})
    raise SystemExit(0 if r.wasSuccessful()else 1)
