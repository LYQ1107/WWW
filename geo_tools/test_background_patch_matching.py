import unittest,numpy as np,torch
from common import ROOT
from gtr.geometry.roi_feature_extractor import PadMapping
from gtr.geometry.background_patch_matcher import background_descriptors,mutual_ratio_matches,validate_candidate

class Tests(unittest.TestCase):
    def test_mutual_and_reverse(self):
        desc=torch.eye(4);points=torch.tensor([[0.,0.],[20.,0.],[0.,20.],[20.,20.]])
        permutation=torch.tensor([2,0,3,1]);shift=torch.tensor([8.,3.])
        a,b=mutual_ratio_matches(desc,desc[permutation],points,points[permutation]+shift)
        self.assertTrue(np.allclose(b-a,shift.numpy()))
        rb,ra=mutual_ratio_matches(desc[permutation],desc,points[permutation]+shift,points)
        self.assertEqual(set(map(tuple,np.c_[a,b])),set(map(tuple,np.c_[ra,rb])))
        repeated=torch.ones(4,4)/2
        self.assertEqual(len(mutual_ratio_matches(repeated,repeated,points,points)[0]),0)

    def test_foreground_and_padding(self):
        mapping=PadMapping(28,14,target=28,patch=14)
        desc,points=background_descriptors(torch.ones(1,3,2,2),mapping,torch.empty(0,4))
        # The official vertical offset is7px: neither 14px patch lies fully
        # inside the content. Padding contamination must be rejected.
        self.assertEqual(len(points),0)
        mapping=PadMapping(28,28,target=28,patch=14)
        desc,points=background_descriptors(torch.ones(1,3,2,2),mapping,torch.tensor([[0.,0.,6.,6.]]))
        self.assertEqual(len(points),3)
        self.assertTrue(torch.isfinite(desc).all())

    def test_independent_pixel_validation(self):
        x,y=np.meshgrid(np.arange(10.,91.,20.),np.arange(10.,91.,20.));p=np.c_[x.ravel(),y.ravel()]
        H=np.array([[1.,0.,4.],[0.,1.,3.],[0.,0.,1.]])
        self.assertTrue(validate_candidate(H,'H',p,p+[4,3],[(100,100)]*2)['valid'])
        self.assertFalse(validate_candidate(H,'H',p[:10],p[:10]+[4,3],[(100,100)]*2)['valid'])
        self.assertFalse(validate_candidate(H,'H',p,p+[40,30],[(100,100)]*2)['valid'])
        self.assertFalse(validate_candidate(np.zeros((3,3)),'F',p,p,[(100,100)]*2)['valid'])
        self.assertFalse(validate_candidate(np.eye(3),'F',p,p,[(100,100)]*2)['valid'])

if __name__=='__main__':unittest.main()
