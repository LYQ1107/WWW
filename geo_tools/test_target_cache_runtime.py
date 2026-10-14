import tempfile,unittest,torch
from pathlib import Path
from common import ROOT
from test_head_runtime import config,save_head
from test_empty_tracking import detections
from gtr.geometry.runtime import GeometryRuntime


class Tests(unittest.TestCase):
    def test_reuse_identity_age_does_not_move_current_geometry_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'head.pt';save_head(path);cfg=config(path);cfg.REUSE_TARGET_FEATURES=True;cfg.UPDATE_INTERVAL=3
            runtime=GeometryRuntime(cfg);records=[dict(width=30,height=20,view_id=v) for v in [1,2]]
            first=[detections(1),detections(1)];runtime.observe(first,records,0)
            second=[detections(1),detections(1)];before=torch.get_rng_state().clone();runtime.observe(second,records,1)
            self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertEqual(runtime.calls,1)
            self.assertTrue(second[1].geo_valid.all());self.assertEqual(second[1].geo_frame.tolist(),[1]);self.assertEqual(second[1].geo_feature_frame.tolist(),[0])
            self.assertTrue(torch.equal(first[1].geo_vector,second[1].geo_vector))
            output=[torch.zeros(1,1)];result=runtime.adapt(output,[first[0],second[1]],1)
            expected=2*torch.tanh(torch.tensor([[.5]]))*20/21
            self.assertTrue(torch.allclose(result[0],expected,atol=1e-6))
            # A stale F at source0 is never retrieved as current geometry1.
            runtime.score.geometry_weight=.5;runtime.geometry[0]=dict(valid=True,F=torch.eye(3).numpy(),inlier_ratio=1.,camera_order=(1,2))
            self.assertTrue(torch.equal(runtime.adapt(output,[first[0],second[1]],1)[0],result[0]))

    def test_cache_oom_is_neutral_and_does_not_reserve_encoder_work(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'head.pt';save_head(path);cfg=config(path);cfg.REUSE_TARGET_FEATURES=True;cfg.UPDATE_INTERVAL=3
            runtime=GeometryRuntime(cfg);records=[dict(width=30,height=20,view_id=v) for v in [1,2]]
            runtime.observe([detections(1),detections(1)],records,0)
            def fail(*args,**kwargs):raise RuntimeError('injected cache OOM')
            runtime.target_cache.reuse=fail;current=[detections(1),detections(1)];runtime.observe(current,records,1)
            self.assertEqual(runtime.failed,1);self.assertEqual(runtime.calls,1);self.assertFalse(current[0].geo_valid.any());self.assertFalse(runtime.target_cache.entries)

    def test_off_and_zero_ignore_cache_head(self):
        cfg=config('/not/present.pt');cfg.REUSE_TARGET_FEATURES=True;cfg.FEATURE_WEIGHT=0
        runtime=GeometryRuntime(cfg);self.assertIsNone(runtime.target_cache)
        data=[detections(1),detections(1)];output=[torch.ones(1,1)]
        self.assertIs(runtime.adapt(output,data,1),output)

    def test_prefix_output_and_call_counts_with_integrated_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'head.pt';save_head(path)
            def run(length):
                cfg=config(path);cfg.REUSE_TARGET_FEATURES=True;cfg.UPDATE_INTERVAL=3
                runtime=GeometryRuntime(cfg);rows=[]
                for frame in range(length):
                    current=[detections(1),detections(1)];runtime.observe(current,[dict(width=30,height=20,view_id=v) for v in [1,2]],frame)
                    rows.append([(inst.geo_vector.tolist(),inst.geo_valid.tolist(),inst.geo_frame.tolist(),inst.geo_feature_frame.tolist()) for inst in current])
                return rows,runtime.calls
            short,calls=run(8);long,_=run(12);self.assertEqual(short,long[:8]);self.assertEqual(calls,3)


if __name__=='__main__':unittest.main()
