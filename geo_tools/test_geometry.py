import unittest,numpy as np
from common import ROOT
from gtr.geometry.geometry_alignment import *
from gtr.geometry.geometry_confidence import estimate_background

class GeometryTests(unittest.TestCase):
    def test_projection_and_propagation(self):
        rng=np.random.default_rng(42);X=rng.normal(size=(100,3));X[:,2]+=6
        K=np.array([[400.,0,250],[0,400,250],[0,0,1]])
        E1=np.c_[np.eye(3),np.zeros(3)];E2=np.c_[np.eye(3),[-.7,.1,0]]
        def proj(E):
            v=(X@E[:,:3].T+E[:,3])@K.T;return v[:,:2]/v[:,2:]
        p1,p2=proj(E1),proj(E2);F=fundamental_from_cameras(E1,E2,K,K)
        self.assertLess(symmetric_epipolar_distance(F,p1,p2).max(),1e-8)
        self.assertLess(symmetric_epipolar_distance(F.T,p2,p1).max(),1e-8)
        H1=np.array([[1.1,.03,12],[.01,.9,-4],[.0001,0,1]]);H2=np.array([[.9,-.02,-15],[.02,1.2,6],[0,.0001,1]])
        def transform(p,H):
            q=np.c_[p,np.ones(len(p))]@H.T;return q[:,:2]/q[:,2:]
        Fn=propagate_fundamental(F,H1,H2)
        self.assertLess(symmetric_epipolar_distance(Fn,transform(p1,H1),transform(p2,H2)).max(),1e-8)
    def test_degenerate(self):
        b=np.array([[0.,0.,1.,1.]])
        self.assertEqual(soft_box_evidence(np.full((3,3),np.nan),b,b,True).sum(),0)
        self.assertEqual(soft_box_evidence(None,b,b,False).sum(),0)
        self.assertFalse(estimate_background(np.zeros((128,128),np.uint8),np.zeros((128,128),np.uint8))['valid'])
        self.assertEqual(soft_box_evidence(np.eye(3),np.empty((0,4)),b,True).shape,(1,0))
if __name__=='__main__':unittest.main()
