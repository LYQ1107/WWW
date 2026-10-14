import tempfile
import unittest
from pathlib import Path
from common import SOURCE_COMMIT,ROOT,sha256
from audit_suite_settings import signature


class Tests(unittest.TestCase):
    def run_info(self,path):
        return dict(commit=SOURCE_COMMIT,checkpoint_sha256='model',config_sha256='config',annotation_sha256='data',
                    seed=14,causal_online=True,source_hashes={'gtr/config.py':sha256(ROOT/'gtr/config.py')},
                    geo_options=dict(ENABLED=True,FEATURE_ENABLED=True,FEATURE_WEIGHT=2.,GEOMETRY_ENABLED=True,
                                     GEOMETRY_WEIGHT=0.,CALIBRATION_PATH=str(path),PROJECTION_DIM=-1))

    def test_effective_projection_and_inactive_geometry_are_compared_semantically(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'calibration.json';path.write_text('{}')
            run=self.run_info(path);left,verified=signature(run)
            self.assertTrue(verified)
            run['geo_options'].update(PROJECTION_DIM=256,GEOMETRY_METHOD='vggt_camera')
            self.assertEqual(signature(run)[0],left)
            run['geo_options']['PROJECTION_DIM']=0
            self.assertNotEqual(signature(run)[0],left)
            run['geo_options'].update(PROJECTION_DIM=256,GEOMETRY_WEIGHT=.5)
            self.assertNotEqual(signature(run)[0],left)

    def test_hash_spoof_is_rejected_and_unrecorded_defaults_are_not_certified(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'artifact';path.write_text('actual bytes')
            run=self.run_info(path);run['geo_options']['HEAD_PATH']=str(path)
            run['head_checkpoint_sha256']='wrong'
            with self.assertRaisesRegex(ValueError,'head artifact hash mismatch'):signature(run)
            run['head_checkpoint_sha256']=sha256(path)
            run['source_hashes']['gtr/config.py']='different uncommitted snapshot'
            self.assertFalse(signature(run)[1])
            run['geo_options']['ENABLED']=False
            path.unlink()
            self.assertFalse(signature(run)[0]['feature_active'])


if __name__=='__main__':unittest.main()
