import ast, unittest, numpy as np, torch
from pathlib import Path
from common import ROOT
from gtr.geometry.roi_feature_extractor import PadMapping,patch_map,pool_roi

class Contracts(unittest.TestCase):
    def test_order(self):
        # Execute only pure order helper; no Detectron model or historical caches.
        tree=ast.parse((ROOT/'gtr/evaluation/mot_evaluation.py').read_text()); fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='align_visiontrack_inputs_to_outputs')
        ns={}; exec(compile(ast.Module(body=[fn],type_ignores=[]),'order','exec'),ns); f=ns[fn.name]
        rows=[dict(image_id=v*3+t,view_num=2) for v in range(2) for t in range(3)]
        self.assertEqual([r['image_id'] for r in f('VISION_train',rows,[None]*6)],[0,3,1,4,2,5])
        with self.assertRaises(ValueError):f('VISION_train',rows,[None]*5)
    def test_pad_roundtrip(self):
        for w,h in [(1920,1080),(1080,1920),(640,640),(1231,977)]:
            m=PadMapping(w,h); boxes=torch.tensor([[0.,0.,w,h],[w*.2,h*.3,w*.4,h*.8]])
            self.assertTrue(torch.allclose(m.inverse_boxes(m.boxes(boxes)),boxes,atol=.0002))
    def test_roi_coordinate(self):
        h,w=37,37
        fm=torch.stack(torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')).float()[None]
        vec,count=pool_roi(fm,torch.tensor([[14.,28.,28.,42.]]),'roi_align')
        expected=torch.tensor([[2.,1.]])
        self.assertTrue(torch.allclose(vec,torch.nn.functional.normalize(expected,dim=-1),atol=1e-5))
        self.assertEqual(count.tolist(),[1])
    def test_special_tokens_and_small(self):
        patch=torch.arange(4*3).reshape(4,3).float(); t=torch.cat([torch.ones(5,3)*999,patch])
        fm=patch_map(t,5,(28,28)); self.assertEqual(fm[0,:,0,0].tolist(),[0,1,2])
        v,n=pool_roi(fm,torch.tensor([[1.,1.,2.,2.]]),'mean');self.assertEqual(n.tolist(),[0]); self.assertEqual(v.abs().sum(),0)

if __name__=='__main__': unittest.main()
