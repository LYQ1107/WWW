"""The capacity control reuses only native detector ReID, with no new encoder."""
import unittest, torch
from common import ROOT
from detectron2.config import get_cfg
from gtr.config import add_gtr_config
from gtr.geometry.runtime import GeometryRuntime
from test_empty_tracking import detections

class Tests(unittest.TestCase):
    def test_native_features_and_real_residual(self):
        cfg = get_cfg();add_gtr_config(cfg);geo = cfg.MODEL.GEO
        geo.ENABLED = True;geo.FEATURE_ENABLED = True;geo.FEATURE_WEIGHT = .5
        geo.FEATURE_BACKBONE = 'gmt';geo.FEATURE_LAYERS = [-1];geo.PROJECTION_DIM = 0
        geo.POOLING = 'reid';geo.SYMMETRIZE_VIEW_ORDER = False;geo.UPDATE_INTERVAL = 1
        geo.CALIBRATION_PATH = str(ROOT / 'manifests/calibration_gmt_reid_absolute_train_v1.json')
        runtime = GeometryRuntime(geo)
        refs, query = detections(1), detections(1)
        original = refs.reid_features.clone()
        before_rng = torch.get_rng_state().clone()
        runtime._observe_impl([refs, query], [dict(width=30,height=20,view_id=v) for v in [1,2]], 0)
        result = runtime.adapt([torch.zeros(1,1)], [refs, query], 1)[0]
        self.assertGreater(float(result[0,0]), 0.)
        self.assertLessEqual(float(result[0,0]), .5)
        self.assertTrue(torch.equal(original, refs.reid_features))
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertTrue(torch.allclose(refs.geo_vector.norm(dim=-1), torch.ones(1)))
        self.assertIsNone(runtime.vggt)
        self.assertEqual(runtime.vggt_calls + runtime.dinov2_calls, 0)

if __name__ == '__main__':
    unittest.main()
