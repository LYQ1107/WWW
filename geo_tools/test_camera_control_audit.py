import hashlib
import unittest
from unittest.mock import patch
from audit_camera_control import verify_pure_gmt


class Tests(unittest.TestCase):
    def test_legacy_requires_exact_captured_driver_and_zero_calls(self):
        digest='38b783425133ba41875c4596015e7ae520c22ed88282cc068ea1879f7e0f086e'
        manifest=dict(feature_weight=0,commit='legacy',source_hashes={'geo_tools/run_online.py':digest},geo_calls=0,vggt_calls=0)
        with patch('audit_camera_control.subprocess.check_output',return_value=b'inspected driver'),patch('audit_camera_control.hashlib.sha256') as sha:
            sha.return_value.hexdigest.return_value=digest
            self.assertEqual(verify_pure_gmt(manifest),'captured_legacy_driver_hash_and_zero_calls')
            for changed in [dict(geo_calls=1),dict(vggt_calls=1),dict(feature_weight=1),dict(source_hashes={'geo_tools/run_online.py':'unknown'})]:
                with self.assertRaises(ValueError):verify_pure_gmt({**manifest,**changed})
            sha.return_value.hexdigest.return_value='unknown'
            with self.assertRaises(ValueError):verify_pure_gmt(manifest)

    def test_recorded_active_geometry_rejected(self):
        with self.assertRaises(ValueError):verify_pure_gmt(dict(feature_weight=0,geo_options={'GEOMETRY_WEIGHT':.5}))


if __name__=='__main__':unittest.main()
