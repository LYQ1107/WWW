import unittest
from unittest.mock import patch
from audit_dense_geometry_ablation import shared_options


class Tests(unittest.TestCase):
    def test_only_geometry_weight_paths_and_inactive_legacy_cache_normalize(self):
        reference=dict(commit='old',source_hashes={'gtr/config.py':'digest'},geo_options=dict(GEOMETRY_WEIGHT=.5,CALIBRATION_PATH='/old/cal',FEATURE_WEIGHT=2))
        feature={**reference,'geo_options':dict(GEOMETRY_WEIGHT=0,CALIBRATION_PATH='/new/cal',FEATURE_WEIGHT=2,REUSE_TARGET_FEATURES=False,TARGET_CACHE_MAX_TARGETS=128)}
        with patch('audit_dense_geometry_ablation.geo_defaults',return_value=({},'digest')),patch('audit_dense_geometry_ablation.sha256',return_value='same-bytes'):
            self.assertEqual(shared_options(reference),shared_options(feature))
            for changed in [dict(FEATURE_WEIGHT=1),dict(UNKNOWN_ACTIVE_OPTION=1)]:
                self.assertNotEqual(shared_options(reference),shared_options({**feature,'geo_options':{**feature['geo_options'],**changed}}))
            with self.assertRaises(ValueError):shared_options({**feature,'geo_options':{**feature['geo_options'],'REUSE_TARGET_FEATURES':True}})
            with self.assertRaises(ValueError):shared_options({**reference,'source_hashes':{'gtr/config.py':'wrong'}})

    def test_calibration_content_difference_is_retained(self):
        run=dict(commit='old',source_hashes={'gtr/config.py':'digest'},geo_options=dict(GEOMETRY_WEIGHT=.5,CALIBRATION_PATH='/cal'))
        with patch('audit_dense_geometry_ablation.geo_defaults',return_value=({},'digest')),patch('audit_dense_geometry_ablation.sha256',side_effect=['before','after']):
            self.assertNotEqual(shared_options(run),shared_options(run))


if __name__=='__main__':unittest.main()
