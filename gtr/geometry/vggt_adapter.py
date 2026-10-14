"""Freeze VGGT, retain original layer indices, capture embedded DINO tokens once."""
import torch
from .roi_feature_extractor import patch_map, pool_roi, compress, PadMapping

class FrozenVGGT:
    def __init__(self, checkpoint, device='cuda', layers=(4,11,17,23)):
        from vggt.models.vggt import VGGT
        self.device=device; self.layers=tuple(layers)
        # Load the complete official architecture, then keep only required heads.
        model=VGGT()
        state=torch.load(checkpoint,map_location='cpu')
        model.load_state_dict(state,strict=True)
        self.parameters=sum(p.numel() for p in model.parameters())
        model.point_head=None; model.depth_head=None; model.track_head=None
        self.model=model.to(device).eval().requires_grad_(False)
        self.dtype=torch.bfloat16 if torch.cuda.get_device_capability()[0]>=8 else torch.float16
        self._dino=None
        self._hook=self.model.aggregator.patch_embed.register_forward_hook(self._capture)

    def _capture(self,module,inputs,output):
        self._dino=output['x_norm_patchtokens'] if isinstance(output,dict) else output

    @torch.inference_mode()
    def extract(self, images, boxes_by_view, modes=('mean','center','roi_align'), dim=256, camera=False):
        if images.ndim==4: images=images[None]
        images=images.to(self.device)
        self._dino=None
        with torch.autocast('cuda',dtype=self.dtype): tokens,start=self.model.aggregator(images)
        views=images.shape[1]; hw=images.shape[-2:]
        if len(boxes_by_view)!=views: raise ValueError('view mismatch')
        dino=self._dino.reshape(1,views,-1,self._dino.shape[-1])
        maps={'dino_internal':(dino,0)}
        for layer in self.layers:
            if layer>=len(tokens) or tokens[layer] is None: raise ValueError(f'uncached layer {layer}')
            maps[f'layer_{layer}']=(tokens[layer],start)
        result={}; coverage={}
        for key,(value,idx) in maps.items():
            for mode in modes:
                vectors=[]
                for v,boxes in enumerate(boxes_by_view):
                    fm=patch_map(value[0,v],idx,hw)
                    pooled,count=pool_roi(fm,boxes.to(self.device),mode)
                    vectors.append(compress(pooled,dim).cpu()); coverage[v]=count.cpu()
                result[f'{key}/{mode}']=vectors
        shapes={str(i):None if x is None else list(x.shape) for i,x in enumerate(tokens)}
        pose=None
        if camera:
            # Camera head expects FP32, following official forward.
            with torch.autocast('cuda',enabled=False): pose=self.model.camera_head(tokens)[-1].float().cpu()
        self._dino=None
        return result,coverage,dict(layer_shapes=shapes,patch_start_idx=start,pose=pose)
