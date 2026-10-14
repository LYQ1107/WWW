import unittest,numpy as np
from common import ROOT
from gtr.geometry.geometry_alignment import project_points,symmetric_transfer_distance,propagate_plane_homography,soft_plane_evidence

class PlaneTests(unittest.TestCase):
    def test_camera_order_and_temporal_direction(self):
        rng=np.random.default_rng(42);p1=rng.uniform(0,500,(50,2));H=np.array([[1.1,.05,20],[.02,.95,8],[.0001,.0002,1.]])
        p2=project_points(H,p1)
        self.assertLess(symmetric_transfer_distance(H,p1,p2).max(),1e-8)
        self.assertLess(symmetric_transfer_distance(np.linalg.inv(H),p2,p1).max(),1e-8)
        A1=np.array([[.9,-.01,6],[.01,1.1,-4],[0,.0001,1.]])
        A2=np.array([[1.05,.02,-12],[.03,.98,13],[.0001,0,1.]])
        nextH=propagate_plane_homography(H,A1,A2)
        self.assertLess(symmetric_transfer_distance(nextH,project_points(A1,p1),project_points(A2,p2)).max(),1e-8)
    def test_invalid_neutral(self):
        boxes=np.array([[0.,0.,10.,20.]])
        for matrix in [None,np.zeros((3,3)),np.full((3,3),np.nan)]:self.assertEqual(soft_plane_evidence(matrix,boxes,boxes,True).sum(),0.)
        self.assertEqual(soft_plane_evidence(np.eye(3),np.empty((0,4)),boxes,True).shape,(1,0))
if __name__=='__main__':unittest.main()
