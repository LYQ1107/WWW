import unittest
from omega_corr_eval_helpers import common_candidates,certified_change,top_reference,cluster_interval

class Tests(unittest.TestCase):
    def test_common_query_is_not_common_candidate_evidence(self):
        # A method with the only positive available cannot establish a feature
        # advantage over another method whose positive has no valid token.
        omega={1:.9,2:.2};dino={2:.8}
        common=common_candidates([omega,dino]);self.assertEqual(common,{2});self.assertNotIn(1,common)

    def test_unknown_not_wrong_and_exact_tie_uses_native_order(self):
        self.assertFalse(certified_change('UNKNOWN_REFERENCE_HISTORY','CORRECT')['correction'])
        self.assertFalse(certified_change('CORRECT','MIXED_REFERENCE_HISTORY')['harm'])
        self.assertEqual(top_reference({11:.5,12:.5},{11:3,12:7},[7,3]),12)

    def test_bootstrap_clusters_require_distinct_scenes(self):
        self.assertIsNone(cluster_interval([(4,1,1),(4,2,1)])['lower95'])
        r=cluster_interval([(4,1,1),(9,1,1)],repeats=100)
        self.assertEqual(r['lower95'],1);self.assertEqual(r['person_clusters'],2)

if __name__=='__main__':unittest.main()
