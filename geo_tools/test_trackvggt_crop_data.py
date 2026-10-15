import unittest
import numpy as np
import torch
from PIL import Image
from trackvggt_crop_data import crop_rectangle,crop_image,crop_target_pool,merge_target_arrays
from prepare_trackvggt_crop_comparison import select_fit


class CropTests(unittest.TestCase):
    def test_native_box_only_clipped_context_and_no_fake_outside_crop(self):
        self.assertEqual(crop_rectangle([0,0,10,20],100,50),[0,0,13,25])
        self.assertIsNone(crop_rectangle([110,10,120,20],100,50))
        with self.assertRaises(ValueError):crop_rectangle([1,2,1,3],100,50)

    def test_pad224_retains_original_crop_coordinates(self):
        image=Image.new('RGB',(200,100),(255,0,0));rect=[10,20,110,70]
        values,mapping=crop_image(image,rect)
        self.assertEqual(tuple(values.shape),(1,1,3,224,224));self.assertEqual(mapping.resized,(224,112))
        self.assertTrue(torch.equal(values[0,0,:,0,0],torch.ones(3)))
        box=torch.tensor([[0.,0.,100.,50.]])
        self.assertTrue(torch.allclose(mapping.inverse_boxes(mapping.boxes(box)),box))

    def test_local_pool_bounds_and_real_original_coordinate_grid(self):
        layers={k:torch.arange(256*8,dtype=torch.float32).reshape(1,1,256,8)+k for k in (4,11,17,23)}
        attention=torch.arange(256,dtype=torch.float32).reshape(1,1,256)
        arrays,counts=crop_target_pool(layers,attention,dict(box=[30.,40.,70.,80.]),[20,30,80,90])
        self.assertGreater(counts['raw_patch_centers'],0);self.assertLessEqual(counts['valid_pool_tokens'],32)
        self.assertTrue(np.all(arrays['xy']>=[20,30]));self.assertTrue(np.all(arrays['xy']<[80,90]))
        self.assertEqual(arrays['roi'].shape,(4,8));self.assertEqual(arrays['layer_4'].dtype,np.float16)

    def test_selection_ignores_GT_errors_and_patch_bins(self):
        items=[dict(scene=s,key=[s,i,0]) for s in (4,9,24) for i in range(40)]
        before=[r['key'] for r in select_fit(items,30)]
        for i,r in enumerate(items):r.update(query_gt_identity=i,known_wrong=bool(i%2),patch_bin='0')
        self.assertEqual(before,[r['key'] for r in select_fit(items,30)])
        self.assertEqual([sum(k[0]==s for k in before) for s in (4,9,24)],[10,10,10])

    def test_combination_does_not_change_inputs_or_bias_attention_by_raster_size(self):
        full=dict(attention=np.array([.001,.002]),xy=np.array([[1.,2.],[2.,3.]]),inside=np.array([True,False]),patch_indices=np.array([1,2]),roi=np.ones((4,3)))
        local=dict(attention=np.array([.1,.2]),xy=np.array([[1.,2.],[2.,3.]]),inside=np.array([True,False]),patch_indices=np.array([1,2]),roi=np.ones((4,3)))
        for k in (4,11,17,23):full['layer_'+str(k)]=np.ones((2,3));local['layer_'+str(k)]=np.ones((2,3))*2
        both=merge_target_arrays(full,local,'combined')
        self.assertEqual(both['attention'].tolist(),[0.,1.,0.,1.]);self.assertEqual(both['layer_4'].shape,(4,3))
        self.assertEqual(full['attention'].tolist(),[.001,.002]);self.assertEqual(local['layer_4'].shape,(2,3))


if __name__=='__main__':unittest.main()
