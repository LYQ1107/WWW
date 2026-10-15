import unittest
from unittest.mock import patch
from audit_tracking_retrieval_prefix import verify_tracking_options


class Tests(unittest.TestCase):
    def test_verified_legacy_no_cache_is_false_but_unknown_or_changed_options_fail(self):
        reference=dict(commit='old',source_hashes={'gtr/config.py':'digest'},geo_options={'FEATURE_WEIGHT':2})
        info=dict(tracking_options={'FEATURE_WEIGHT':2,'REUSE_TARGET_FEATURES':False})
        with patch('audit_tracking_retrieval_prefix.geo_defaults',return_value=({},'digest')):
            self.assertEqual(verify_tracking_options(info,reference),'digest')
            for changed in [{'REUSE_TARGET_FEATURES':True},{'FEATURE_WEIGHT':1},{'UNKNOWN_OPTION':False}]:
                with self.assertRaises(ValueError):verify_tracking_options(dict(tracking_options={**info['tracking_options'],**changed}),reference)
            with self.assertRaises(ValueError):verify_tracking_options(info,dict(reference,source_hashes={'gtr/config.py':'wrong'}))


if __name__=='__main__':unittest.main()
