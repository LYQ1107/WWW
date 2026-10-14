"""Exact official VGGT pad-mode coordinate mapping and target-only pooling."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

@dataclass(frozen=True)
class PadMapping:
    width: int
    height: int
    target: int = 518
    patch: int = 14

    @property
    def resized(self):
        if self.width >= self.height:
            return self.target, round(self.height * self.target / self.width / self.patch) * self.patch
        return round(self.width * self.target / self.height / self.patch) * self.patch, self.target

    def boxes(self, xyxy):
        w,h = self.resized
        scale = xyxy.new_tensor([w/self.width, h/self.height, w/self.width, h/self.height])
        offset = xyxy.new_tensor([(self.target-w)//2,(self.target-h)//2]*2)
        return xyxy * scale + offset

    def inverse_boxes(self, xyxy):
        w,h = self.resized
        scale = xyxy.new_tensor([w/self.width, h/self.height, w/self.width, h/self.height])
        offset = xyxy.new_tensor([(self.target-w)//2,(self.target-h)//2]*2)
        return (xyxy-offset)/scale

def patch_map(tokens, patch_start_idx, image_hw, patch=14):
    """Accept one view [P,C]; strip special tokens with explicit index."""
    h,w = image_hw[0]//patch, image_hw[1]//patch
    tokens = tokens[patch_start_idx:]
    if tokens.shape[0] != h*w: raise ValueError(f'patch count mismatch: {tokens.shape[0]} != {h*w}')
    return tokens.transpose(0,1).reshape(1,-1,h,w).float()

def pool_roi(feature_map, boxes, mode='roi_align', patch=14):
    """Returns vectors and actual patch-center coverage (including zero)."""
    _,c,h,w=feature_map.shape
    if len(boxes)==0: return feature_map.new_empty((0,c)), boxes.new_empty((0,))
    boxes=boxes.float().clone(); boxes[:,0::2].clamp_(0,w*patch); boxes[:,1::2].clamp_(0,h*patch)
    ys=(torch.arange(h,device=boxes.device)+.5)*patch
    xs=(torch.arange(w,device=boxes.device)+.5)*patch
    coverage=[]; pooled=[]
    for b in boxes:
        hit=(ys[:,None]>=b[1])&(ys[:,None]<b[3])&(xs[None,:]>=b[0])&(xs[None,:]<b[2])
        coverage.append(hit.sum())
        if mode=='mean':
            if hit.any(): pooled.append(feature_map[0,:,hit].mean(-1))
            else: pooled.append(feature_map.new_zeros(c))
        elif mode=='center':
            cx=((b[0]+b[2])/(2*patch)).long().clamp(0,w-1)
            cy=((b[1]+b[3])/(2*patch)).long().clamp(0,h-1)
            pooled.append(feature_map[0,:,max(0,int(cy)-1):min(h,int(cy)+2),max(0,int(cx)-1):min(w,int(cx)+2)].mean((-1,-2)))
        elif mode!='roi_align': raise ValueError(mode)
    if mode=='roi_align':
        # aligned=True maps image patch centers at 7px to feature sample index 0.
        values=roi_align(feature_map,[boxes],output_size=(3,3),spatial_scale=1/patch,sampling_ratio=2,aligned=True).mean((-1,-2))
    else: values=torch.stack(pooled)
    return F.normalize(values,dim=-1,eps=1e-8),torch.stack(coverage)

def compress(vectors, dim=256, seed=20261014):
    """Fixed random projection. Pool/normalize before projection; no dense-token caching."""
    generator=torch.Generator(device='cpu').manual_seed(seed)
    projection=torch.randn(vectors.shape[-1],dim,generator=generator,dtype=torch.float32)/dim**.5
    return F.normalize(vectors.float()@projection.to(vectors.device),dim=-1,eps=1e-8)
