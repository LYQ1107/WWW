import tempfile,unittest,torch
from pathlib import Path
from common import ROOT
from detectron2.config import get_cfg
from gtr.config import add_gtr_config
from gtr.geometry.lightweight_head import LowRankIdentityHead
from gtr.geometry.runtime import GeometryRuntime
from test_empty_tracking import detections

def config(path):
    cfg=get_cfg();add_gtr_config(cfg);g=cfg.MODEL.GEO
    g.ENABLED=True;g.FEATURE_ENABLED=True;g.FEATURE_WEIGHT=2.;g.MAX_RESIDUAL=2.
    g.FEATURE_BACKBONE='gmt';g.FEATURE_LAYERS=[-1];g.PROJECTION_DIM=0
    g.POOLING='reid';g.SYMMETRIZE_VIEW_ORDER=False;g.UPDATE_INTERVAL=1;g.HEAD_PATH=str(path)
    return g

def save_head(path,gmt_dim=8):
    settings=dict(gmt_dim=gmt_dim,foundation_dim=0,hidden_dim=16,output_dim=4,symmetric=False)
    model=LowRankIdentityHead(**settings)
    with torch.no_grad():model.scale.fill_(1.);model.bias.fill_(-.5)
    torch.save(dict(settings=settings,state_dict=model.state_dict(),source_split='association_train',backbone='gmt',layers=[-1],pooling='reid',projection_dim=0,feature_weight=2.,max_residual=2.),path)

class Tests(unittest.TestCase):
    def test_actual_head_runtime_projection_and_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'head.pt';save_head(path);runtime=GeometryRuntime(config(path))
            refs,query=detections(1),detections(1);before=torch.get_rng_state().clone()
            records=[dict(width=30,height=20,view_id=v) for v in [1,2]]
            runtime.observe([refs,query],records,0)
            self.assertEqual(refs.geo_vector.shape,(1,4));self.assertTrue(torch.equal(before,torch.get_rng_state()))
            self.assertTrue(all(not p.requires_grad for p in runtime.head.parameters()))
            base=[torch.zeros(1,1)];result=runtime.adapt(base,[refs,query],1)
            self.assertTrue(torch.allclose(result[0],2*torch.tanh(torch.tensor([[.5]])),atol=1e-6))

    def test_off_ignores_unavailable_checkpoint(self):
        cfg=config('/nonexistent/head.pt');cfg.FEATURE_WEIGHT=0.
        runtime=GeometryRuntime(cfg);outputs=[torch.randn(1,1)]
        self.assertIs(runtime.adapt(outputs,[detections(1),detections(1)],1),outputs)

    def test_partial_projection_failure_restores_uniform_field_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'head.pt';save_head(path,gmt_dim=99);runtime=GeometryRuntime(config(path))
            refs,query=detections(1),detections(1)
            runtime.observe([refs,query],[dict(width=30,height=20,view_id=v) for v in [1,2]],0)
            self.assertEqual(runtime.failed,1);self.assertFalse(query.geo_valid.any())
            self.assertEqual(refs.geo_vector.shape,(1,4));self.assertEqual(query.geo_vector.shape,(1,4))
            base=[torch.randn(1,1)]
            self.assertTrue(torch.equal(runtime.adapt(base,[refs,query],1)[0],base[0]))

if __name__=='__main__':unittest.main()
