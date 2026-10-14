import unittest,numpy as np
from common import ROOT
from gtr.geometry.camera_geometry import validate_camera_fundamental
from gtr.geometry.geometry_alignment import symmetric_epipolar_distance
from gtr.geometry.roi_feature_extractor import PadMapping

class Tests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(20261014);world=rng.normal(size=(100,3));world[:,2]+=6
        K=np.repeat(np.array([[[400.,0.,259.],[0.,400.,259.],[0.,0.,1.]]]),2,axis=0)
        E=np.array([np.c_[np.eye(3),np.zeros(3)],np.c_[np.eye(3),[-.7,.1,0.]]])
        mappings=[PadMapping(1920,1080),PadMapping(1280,960)]
        points=[]
        for e,k,m in zip(E,K,mappings):
            x=(world@e[:,:3].T+e[:,3])@k.T;p=x[:,:2]/x[:,2:]
            w,h=m.resized;points.append((p-[(m.target-w)//2,(m.target-h)//2])/[w/m.width,h/m.height])
        return E,K,mappings,dict(points1=points[0],points2=points[1],coverage=.02)

    def test_joint_pose_original_coordinates_and_camera_reverse(self):
        E,K,mappings,bg=self.fixture();result=validate_camera_fundamental(E,K,mappings,bg)
        self.assertTrue(result['valid'])
        self.assertLess(symmetric_epipolar_distance(result['F'],bg['points1'],bg['points2']).max(),1e-7)
        rev=validate_camera_fundamental(E[::-1],K[::-1],mappings[::-1],dict(points1=bg['points2'],points2=bg['points1'],coverage=.02))
        self.assertTrue(rev['valid'])
        self.assertLess(symmetric_epipolar_distance(rev['F'],bg['points2'],bg['points1']).max(),1e-7)

    def test_degenerate_and_insufficient_reference(self):
        E,K,mappings,bg=self.fixture()
        self.assertFalse(validate_camera_fundamental(np.repeat(E[:1],2,axis=0),K,mappings,bg)['valid'])
        bg['points1']=bg['points1'][:5];bg['points2']=bg['points2'][:5]
        self.assertFalse(validate_camera_fundamental(E,K,mappings,bg)['valid'])

if __name__=='__main__':unittest.main()
