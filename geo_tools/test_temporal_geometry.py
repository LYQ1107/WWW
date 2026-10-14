import unittest,numpy as np
from common import ROOT
from gtr.geometry.temporal_geometry import TemporalGeometryConsumer
from gtr.geometry.geometry_alignment import fundamental_from_cameras,symmetric_epipolar_distance,project_points

class Tests(unittest.TestCase):
    def test_sequential_warp_direction_and_completion(self):
        rng=np.random.default_rng(42);X=rng.normal(size=(40,3));X[:,2]+=6
        K=np.array([[300.,0,200],[0,300,150],[0,0,1.]])
        E1=np.c_[np.eye(3),np.zeros(3)];E2=np.c_[np.eye(3),[-.8,.1,0]]
        def points(E):
            x=(X@E[:,:3].T+E[:,3])@K.T;return x[:,:2]/x[:,2:]
        p1,p2=points(E1),points(E2);F=fundamental_from_cameras(E1,E2,K,K)
        consumer=TemporalGeometryConsumer(5)
        message=dict(source_frame=0,completed_at=10.,result=dict(F=F,valid=True,camera_order=(1,2)))
        self.assertFalse(consumer.receive(message,0,now=9.))
        self.assertTrue(consumer.receive(message,1,now=11.))
        a=np.array([[1,.01,5],[0,1,-2],[0,0,1.]])
        b=np.array([[1,0,-3],[.01,1,4],[0,0,1.]])
        for frame in [1,2]:consumer.add_motion(frame,[dict(camera=c,valid=True,temporal_H_original=H.tolist(),median_residual_processed_px=.1) for c,H in [(1,a),(2,b)]])
        candidate=consumer.candidate(2,now=11.)
        self.assertLess(symmetric_epipolar_distance(candidate['F'],project_points(a@a,p1),project_points(b@b,p2)).max(),1e-7)
        self.assertFalse(candidate['valid']) # current independent validation required
        self.assertIsNone(consumer.candidate(6,now=11.))

    def test_missing_nan_and_large_residual_disable(self):
        F=np.array([[0.,0.,1.],[0.,0.,0.],[-1.,0.,0.]])
        consumer=TemporalGeometryConsumer(5)
        consumer.receive(dict(source_frame=0,completed_at=1.,result=dict(F=F,valid=True,camera_order=(1,2))),0,now=2.)
        self.assertIsNone(consumer.candidate(1,now=2.))
        consumer.add_motion(1,[dict(camera=c,valid=True,temporal_H_original=np.eye(3).tolist(),median_residual_processed_px=3.) for c in [1,2]])
        self.assertIsNone(consumer.candidate(1,now=2.))
        consumer.add_motion(1,[dict(camera=c,valid=True,temporal_H_original=np.full((3,3),np.nan).tolist(),median_residual_processed_px=.1) for c in [1,2]])
        self.assertIsNone(consumer.candidate(1,now=2.))

if __name__=='__main__':unittest.main()
