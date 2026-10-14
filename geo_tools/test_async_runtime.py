import tempfile,time,unittest,torch,cv2,numpy as np
from pathlib import Path
from unittest.mock import patch
from common import ROOT
from detectron2.config import get_cfg
from gtr.config import add_gtr_config
from gtr.geometry.runtime import GeometryRuntime
from test_empty_tracking import detections

class FakeWorker:
    ready=True;disabled=False;pending=None
    def submit(self,frame,payload):
        self.pending=frame
        self.payload=payload
        assert all(not isinstance(x,torch.Tensor) for boxes in payload['boxes'] for box in boxes for x in box)
        return True
    def poll(self,frame):
        if self.pending is None or frame<=self.pending:return None
        source=self.pending;self.pending=None
        return dict(source_frame=source,completed_at=time.monotonic()-.001,result=dict(F=np.array([[0.,0.,0.],[0.,0.,-1.],[0.,1.,0.]]),valid=True,camera_order=(1,2)))

class FakeMotion:
    valid=True
    def observe(self,records,boxes):
        return 0.,[dict(camera=r['view_id'],valid=self.valid,temporal_H_original=np.eye(3).tolist() if self.valid else None,median_residual_processed_px=0.) for r in records]

class Tests(unittest.TestCase):
    def test_completed_past_geometry_requires_current_validation(self):
        cfg=get_cfg();add_gtr_config(cfg);g=cfg.MODEL.GEO
        g.ENABLED=True;g.GEOMETRY_ENABLED=True;g.GEOMETRY_WEIGHT=.5
        g.ASYNC_WORKER=True;g.ASYNC_GPU='0';g.GEOMETRY_METHOD='vggt_camera';g.UPDATE_INTERVAL=10
        runtime=GeometryRuntime(g);runtime.async_worker=FakeWorker();runtime.motion=FakeMotion()
        x,y=np.meshgrid(np.arange(20.,181.,20.),np.arange(20.,81.,20.));points=np.c_[x.ravel(),y.ravel()]
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'frame.png';cv2.imwrite(str(path),np.zeros((100,200),np.uint8))
            records=[dict(file_name=str(path),width=200,height=100,view_id=v) for v in [1,2]]
            for frame in [0,1,2]:
                refs,query=detections(1),detections(1)
                if frame==2:runtime.motion.valid=False
                with patch('gtr.geometry.runtime.estimate_background',return_value=dict(points1=points,points2=points+[5,0])):
                    runtime.observe([refs,query],records,frame)
                output=runtime.adapt([torch.zeros(1,1)],[refs,query],1)[0]
                if frame==1:
                    self.assertGreater(float(output[0,0]),0.)
                    self.assertEqual(runtime.geometry[1]['source_frame'],0)
                    self.assertTrue(runtime.geometry[1]['valid'])
                else:self.assertEqual(float(output[0,0]),0.)
                self.assertEqual(query.geo_vector.shape,(1,0))
        self.assertEqual(runtime.async_trace[0]['frame'],1)
        self.assertLess(runtime.async_trace[0]['completed_at'],runtime.async_trace[0]['observed_at'])

    def test_invalid_async_configuration_rejected(self):
        cfg=get_cfg();add_gtr_config(cfg);g=cfg.MODEL.GEO
        g.ENABLED=True;g.GEOMETRY_ENABLED=True;g.GEOMETRY_WEIGHT=.5;g.ASYNC_WORKER=True
        with self.assertRaises(ValueError):GeometryRuntime(g)

    def test_fixed_policy_does_not_compute_unused_identity_score(self):
        cfg=get_cfg();add_gtr_config(cfg);g=cfg.MODEL.GEO
        g.ENABLED=True;g.GEOMETRY_ENABLED=True;g.GEOMETRY_WEIGHT=.5
        g.SPARSE_UPDATE_ENABLED=True;g.UPDATE_POLICY='fixed'
        runtime=GeometryRuntime(g)
        self.assertIsNone(runtime.motion)
        def forbidden(*args):raise AssertionError('fixed policy does not need GTA feedback')
        runtime.record_identity_feedback(torch.zeros(1,1),[detections(1),detections(1)],1,forbidden)

if __name__=='__main__':unittest.main()
