"""Independent four-layer DINO control on the same RGB/pad/crop observations."""
import os
from types import SimpleNamespace
import torch
from .trackvggt_features import LAYERS,incoming_attention


class FrozenTrackDINO:
    def __init__(self,checkpoint,device='cuda'):
        os.environ.setdefault('XFORMERS_DISABLED','1')
        from dinov2.hub.backbones import dinov2_vitl14_reg
        model=dinov2_vitl14_reg(pretrained=False)
        model.load_state_dict(torch.load(checkpoint,map_location='cpu'),strict=True)
        self.model=model.to(device).eval().requires_grad_(False);self.device=device;self.calls=0;self.attention=None
        self.dtype=torch.bfloat16 if torch.cuda.get_device_capability()[0]>=8 else torch.float16
        self.mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
        self.std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
        # Explicitly record DINO block23 incoming attention; this is a matched
        # independent-backbone saliency control, not a VGGT attention tensor.
        self.hook=self.model.blocks[23].attn.register_forward_hook(self._capture,with_kwargs=True)

    def _capture(self,module,args,kwargs,output):
        x=args[0];dim=x.shape[-1]//module.num_heads
        wrapper=SimpleNamespace(qkv=module.qkv,num_heads=module.num_heads,head_dim=dim,
            q_norm=torch.nn.Identity(),k_norm=torch.nn.Identity(),rope=None,scale=module.scale)
        self.attention=incoming_attention(wrapper,x)

    @torch.inference_mode()
    def extract(self,images):
        if images.ndim!=5 or images.shape[0]!=1 or images.shape[2]!=3:raise ValueError('expected physical-view RGB batch')
        if not torch.isfinite(images).all() or images.min()<0 or images.max()>1:raise ValueError('RGB must be[0,1] before one DINO normalization')
        images=images[0].to(self.device);self.attention=None
        with torch.backends.cuda.sdp_kernel(enable_flash=False,enable_math=True,enable_mem_efficient=False),torch.backends.cudnn.flags(deterministic=True,benchmark=False):
            with torch.autocast('cuda',dtype=self.dtype):
                values=self.model.get_intermediate_layers((images-self.mean)/self.std,n=LAYERS,norm=True)
        start=1+self.model.num_register_tokens
        layers={layer:value[None] for layer,value in zip(LAYERS,values)}
        attention=self.attention[:,start:][None]
        expected=(images.shape[-2]//14)*(images.shape[-1]//14)
        if len(layers)!=4 or any(value.shape[-2]!=expected or not torch.isfinite(value).all() for value in layers.values()):raise ValueError('DINO layer/patch mismatch')
        if attention.shape!=(1,images.shape[0],expected) or not torch.isfinite(attention).all():raise ValueError('DINO saliency layout mismatch')
        self.calls+=1
        return layers,attention,dict(patch_start_idx=start,layer_ids=list(LAYERS),attention_layer=23,image_hw=list(images.shape[-2:]),backbone='independent_DINOv2_L14_reg4',backend='math_sdpa_deterministic_cudnn')
