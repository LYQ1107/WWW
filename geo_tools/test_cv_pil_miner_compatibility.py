"""Release readiness and GT-free cache joining must not silently invent assets."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_cv_pil_native_handoff import fixture
from probe_cv_pil_miner_compatibility import available_queries,publication_readiness


class TestCompatibility(unittest.TestCase):
    def test_complete_census_without_features_is_not_trainable_release(self):
        r=publication_readiness(dict(status='COMPLETE_12_ASSOCIATION_TRAIN_FULL_CAUSAL_AUDITS',
            scenes=[dict(scene=12,role='validation_diagnostic',features_saved=False)]))
        self.assertFalse(r['trainable_release'])
        self.assertIn('NEW_NATIVE_FEATURE_ASSETS_NOT_EXPORTED',r['missing'])
        self.assertIn('NEW_EXACT_CANDIDATE_SCORE_ASSETS_NOT_EXPORTED',r['missing'])

    def test_availability_does_not_read_gt_or_event_selection(self):
        b=fixture();a=available_queries(b)
        changed=copy.deepcopy(b);changed['payload']['events']='UNREADABLE_OFFLINE_GT_SENTINEL'
        self.assertEqual(a,available_queries(changed))
        self.assertEqual(a[1]['feature_and_native_top8_available'],1)

    def test_missing_assets_are_not_filled_and_future_history_fails(self):
        b=fixture();del b['payload']['observations'][(101,0)]
        rows,counts=available_queries(b);self.assertEqual(rows,[])
        self.assertEqual(counts['top8_history_feature_not_retained'],1)
        b=fixture();b['payload']['observations'][(101,0)]['frame']=11
        with self.assertRaisesRegex(ValueError,'Future'):available_queries(b)


if __name__=='__main__':unittest.main()
