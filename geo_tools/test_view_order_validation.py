import unittest
import torch
from common import ROOT
from gtr.geometry.view_order_validation import reverse_and_restore


class Tests(unittest.TestCase):
    def test_images_rois_pose_and_patch_maps_restore_physical_ids(self):
        for batched in [False,True]:
            images=torch.stack([torch.ones(3,2,2),torch.full((3,2,2),2.)])
            if batched:images=images[None]
            boxes=[torch.tensor([[1.,2,3,4]]),torch.tensor([[5.,6,7,8],[9,10,11,12]])]
            def extract(value,roi,**kwargs):
                observed=value[0] if batched else value
                self.assertEqual(observed[:,0,0,0].tolist(),[2.,1.])
                self.assertTrue(torch.equal(roi[0],boxes[1]))
                return {'layer_17/roi_align':[observed[0,0,:1],observed[1,0,:1]]},{0:'camera2',1:'camera1'},dict(pose=torch.tensor([[[2.],[1.]]]),patch_maps={'l17':['camera2','camera1']})
            features,coverage,meta=reverse_and_restore(extract,images,boxes,camera=True)
            self.assertEqual([float(x[0,0]) for x in features['layer_17/roi_align']],[1.,2.])
            self.assertEqual(coverage,{0:'camera1',1:'camera2'})
            self.assertEqual(meta['pose'].flatten().tolist(),[1.,2.])
            self.assertEqual(meta['patch_maps']['l17'],['camera1','camera2'])


if __name__=='__main__':unittest.main()
