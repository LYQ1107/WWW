import unittest
from types import SimpleNamespace
import torch
from common import ROOT
from test_empty_tracking import detections
from gtr.geometry.pair_score_adapter import PairScoreAdapter
from gtr.geometry.native_prior import install


class Tests(unittest.TestCase):
    def runtime(self,weight=2.):
        cfg=SimpleNamespace(FEATURE_BACKBONE='gmt',HEAD_PATH='',GEOMETRY_WEIGHT=0.,ASYNC_WORKER=False,
                            REUSE_TARGET_FEATURES=False,SPARSE_UPDATE_ENABLED=False,MAX_GEOMETRY_AGE=20)
        runtime=SimpleNamespace(cfg=cfg,active=bool(weight),score=PairScoreAdapter(True,weight,0.,2.),calls=0,seconds=0.)
        install(runtime);return runtime

    def test_prior_is_cross_camera_finite_bounded_and_fresh_without_vectors(self):
        runtime=self.runtime();first,second=detections(1),detections(1)
        records=[dict(view_id=1,width=48,height=32),dict(view_id=2,width=48,height=32)]
        runtime.observe([first,second],records,0)
        self.assertEqual(first.geo_vector.shape,(1,0))
        score=torch.tensor([[.3]]);rng=torch.get_rng_state().clone()
        result=runtime.adapt([score],[first,second],1)[0]
        self.assertTrue(torch.equal(result,score+2))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        second.geo_frame.fill_(21)
        self.assertTrue(torch.equal(runtime.adapt([score],[first,second],1)[0],score))
        second.geo_frame.zero_();second.geo_camera.fill_(1)
        self.assertTrue(torch.equal(runtime.adapt([score],[first,second],1)[0],score))
        second.geo_camera.fill_(2);second.geo_valid.fill_(False)
        self.assertTrue(torch.equal(runtime.adapt([score],[first,second],1)[0],score))

    def test_zero_weight_is_exact_original_object_and_does_not_attach_fields(self):
        runtime=self.runtime(0.);inst=detections(1)
        runtime.observe([inst],[dict(view_id=1,width=48,height=32)],0)
        self.assertFalse(inst.has('geo_frame'))
        outputs=[torch.tensor([[1.]])]
        self.assertIs(runtime.adapt(outputs,[inst],0),outputs)


if __name__=='__main__':unittest.main()
