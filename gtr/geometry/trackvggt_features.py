"""Frozen official VGGT tokens and the exact VGGT-Det incoming-attention statistic.

VGGT-Det 73ea4cf9 captures frame block0 attention, averages heads and query
positions, then removes special KEY tokens. We compute that same statistic
in chunks from the official block's Q/K, including normalization and RoPE.
The original forward and its output tensors are never replaced.
"""
import torch

LAYERS = (4, 11, 17, 23)


def explicit_layer_map(outputs, patch_start_idx, image_hw, compact_layers=None):
    if compact_layers is None:
        if len(outputs) != 24:
            raise ValueError('official output must retain 24 original layer positions')
        result = {layer: outputs[layer] for layer in LAYERS}
    else:
        if tuple(compact_layers) != LAYERS or len(outputs) != len(LAYERS):
            raise ValueError('compact outputs require explicit original layer labels')
        result = dict(zip(compact_layers, outputs))
    count = (image_hw[0] // 14) * (image_hw[1] // 14)
    for value in result.values():
        if value is None or value.ndim != 4 or value.shape[-2] != patch_start_idx + count:
            raise ValueError('invalid layer/special-token/patch layout')
        if not torch.isfinite(value).all():
            raise ValueError('nonfinite frozen features')
    return {layer: value[..., patch_start_idx:, :] for layer, value in result.items()}


def incoming_attention(module, x, pos=None, chunk=128):
    """Mean over attention heads AND all query tokens, retaining key positions."""
    b, n, c = x.shape
    qkv = module.qkv(x).reshape(b,n,3,module.num_heads,module.head_dim).permute(2,0,3,1,4)
    q, k, _ = qkv.unbind(0)
    q, k = module.q_norm(q), module.k_norm(k)
    if module.rope is not None:
        q, k = module.rope(q,pos), module.rope(k,pos)
    total = torch.zeros((b,n),device=x.device,dtype=torch.float32)
    with torch.autocast(x.device.type,enabled=False):
        for first in range(0,n,chunk):
            logits = (q[:,:,first:first+chunk].float() @ k.float().transpose(-2,-1)) * module.scale
            total += logits.softmax(-1).sum((1,2))
    return total / (module.num_heads*n)


class FrozenTrackVGGT:
    def __init__(self, checkpoint, device='cuda'):
        from vggt.models.aggregator import Aggregator
        self.device = device
        self.model = Aggregator()
        full = torch.load(checkpoint,map_location='cpu')
        state = {key.removeprefix('aggregator.'): value for key,value in full.items() if key.startswith('aggregator.')}
        if not state: raise ValueError('missing official aggregator weights')
        self.model.load_state_dict(state,strict=True)
        del state,full
        self.model = self.model.to(device).eval().requires_grad_(False)
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.attention = None
        self.calls = 0
        self.last_input = None
        self.capture_input = False
        self.hook = self.model.frame_blocks[0].attn.register_forward_pre_hook(self._capture,with_kwargs=True)

    def _capture(self,module,args,kwargs):
        x = args[0]
        pos = kwargs.get('pos',args[1] if len(args)>1 else None)
        self.attention = incoming_attention(module,x,pos)
        if self.capture_input: self.last_input = (x.detach().clone(),None if pos is None else pos.detach().clone())

    @torch.inference_mode()
    def extract(self,images):
        if images.ndim != 5 or images.shape[0] != 1 or images.shape[2] != 3:
            raise ValueError('expected [1,physical_views,3,H,W]')
        if not torch.isfinite(images).all() or images.min()<0 or images.max()>1:
            raise ValueError('official RGB [0,1] required; normalization occurs once inside aggregator')
        self.attention = None
        # The bounded real-image repeat test found default FP16 forwards vary
        # even without hooks on this V100/Torch2 stack. This explicit backend
        # produces exact four-layer repeats and matches the pinned Det probe.
        # Scope the choice to the teacher encoder, never the GMT detector.
        with torch.backends.cuda.sdp_kernel(enable_flash=False,enable_math=True,enable_mem_efficient=False),torch.backends.cudnn.flags(deterministic=True,benchmark=False):
            with torch.autocast('cuda',dtype=self.dtype):
                outputs,start = self.model(images.to(self.device))
        layers = explicit_layer_map(outputs,start,images.shape[-2:])
        attention = self.attention[:,start:].reshape(1,images.shape[1],-1)
        if attention.shape[-1] != layers[4].shape[-2] or not torch.isfinite(attention).all():
            raise ValueError('attention orientation or patch shape mismatch')
        self.calls += 1
        return layers,attention,dict(patch_start_idx=start,layer_ids=list(LAYERS),image_hw=list(images.shape[-2:]))
