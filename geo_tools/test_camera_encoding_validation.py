import types,unittest,torch
from common import ROOT
from gtr.geometry.camera_encoding_validation import install_single_image_camera_control


class Tests(unittest.TestCase):
    def test_explicit_view_fraction_and_unchanged_model_parameters(self):
        model=torch.nn.Module();model.roi_heads=torch.nn.Module();model.roi_heads.concat_emb=True
        model.roi_heads.s_t_head=torch.nn.Identity();received=[]
        def inference(batch,view_num,time,view,**kwargs):
            value=torch.zeros(3,11);value[:,0]=.7
            result=model.roi_heads.s_t_head(value);received.append(result.clone());return result
        model.inference=inference;keys=list(model.state_dict());handle=install_single_image_camera_control(model)
        for view in [0,1]:model.inference([{}],2,[1,1,3],view,do_postprocess=False)
        self.assertEqual(received[0][:,-1].tolist(),[0.,0.,0.]);self.assertEqual(received[1][:,-1].tolist(),[.5,.5,.5])
        self.assertTrue(torch.equal(received[0][:,:-1],received[1][:,:-1]));self.assertEqual(keys,list(model.state_dict()))
        with self.assertRaises(RuntimeError):model.roi_heads.s_t_head(torch.zeros(1,11))
        with self.assertRaises(ValueError):model.inference([{}],2,[1,1,3],3)
        handle.remove()


if __name__=='__main__':unittest.main()
