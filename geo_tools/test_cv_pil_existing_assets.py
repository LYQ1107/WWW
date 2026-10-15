from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_cv_pil_native_handoff import fixture
from recover_cv_pil_existing_assets import augment_observations


class TestRecovery(unittest.TestCase):
    def data(self):
        obs=fixture()['payload']['observations']
        detector=[dict(source=[200,0],detection=dict(bbox=[1.,1.,1.,1.])),
                  dict(source=[201,0],detection=dict(bbox=[3.,1.,1.,1.]))]
        meta=[dict(image_id=200,bbox_xyxy=[1.,1.,2.,2.],raw_frame=10,camera=2),
              dict(image_id=201,bbox_xyxy=[3.,1.,4.,2.],raw_frame=11,camera=2)]
        return obs,detector,meta,np.ones((2,1152),dtype=np.float16)

    def test_exact_join_preserves_old_features_and_recovers_without_gt(self):
        obs,det,meta,x=self.data();merged,c=augment_observations(obs,det,meta,x)
        self.assertEqual(c['common_exact_features'],1);self.assertEqual(c['added_native_vectors'],1)
        self.assertNotIn((201,0),obs);self.assertIn((201,0),merged)

    def test_source_mismatch_or_ambiguous_box_fails_closed(self):
        obs,det,meta,x=self.data();x[0,0]=2.
        with self.assertRaisesRegex(ValueError,'differs'):augment_observations(obs,det,meta,x)
        obs,det,meta,x=self.data();det.append(det[0])
        with self.assertRaisesRegex(ValueError,'Ambiguous'):augment_observations(obs,det,meta,x)


if __name__=='__main__':unittest.main()
