"""Independent frozen DINOv2 control with identical RGB/pad/ROI coordinates."""
import torch
from .roi_feature_extractor import patch_map,pool_roi,compress

class FrozenDINOv2:
    def __init__(self,checkpoint,device='cuda'):
        from dinov2.hub.backbones import dinov2_vitl14_reg
        model=dinov2_vitl14_reg(pretrained=False)
        model.load_state_dict(torch.load(checkpoint,map_location='cpu'),strict=True)
        self.model=model.to(device).eval().requires_grad_(False);self.device=device
        self.dtype=torch.bfloat16 if torch.cuda.get_device_capability()[0]>=8 else torch.float16
        self.mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
        self.std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]

    @torch.inference_mode()
    def extract(self,images,boxes_by_view,modes=('roi_align',),dim=256):
        images=images.to(self.device)
        with torch.autocast('cuda',dtype=self.dtype):tokens=self.model.forward_features((images-self.mean)/self.std)['x_norm_patchtokens']
        result={};coverage={}
        for mode in modes:
            vectors=[]
            for v,boxes in enumerate(boxes_by_view):
                fm=patch_map(tokens[v],0,images.shape[-2:]);pooled,count=pool_roi(fm,boxes.to(self.device),mode)
                vectors.append(compress(pooled,dim).cpu());coverage[v]=count.cpu()
            result[f'layer_-1/{mode}']=vectors
        return result,coverage,dict(backbone='independent_dinov2',patch_start_idx=0)
