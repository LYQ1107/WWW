"""Deterministic 2D box/history adaptation of VGGT-Det AG, without 3D FPS."""
import torch
from torch import nn
from torch.nn import functional as F


def normalize_attention(values):
    if not torch.isfinite(values).all():raise ValueError('nonfinite attention')
    if not len(values):return values
    span=values.max()-values.min()
    return (values-values.min())/span.clamp_min(1e-8)


def patch_regions(mapping,box,device='cpu',context_scale=1.5):
    size=mapping.target//mapping.patch
    yy,xx=torch.meshgrid(torch.arange(size,device=device),torch.arange(size,device=device),indexing='ij')
    xy=torch.stack((xx.flatten()+.5,yy.flatten()+.5),-1)*mapping.patch
    rw,rh=mapping.resized;left=(mapping.target-rw)//2;top=(mapping.target-rh)//2
    valid=(xy[:,0]>=left)&(xy[:,0]<left+rw)&(xy[:,1]>=top)&(xy[:,1]<top+rh)
    original=mapping.inverse_boxes(torch.cat((xy,xy),dim=-1))[:,:2]
    box=torch.as_tensor(box,device=device,dtype=torch.float32)
    center=(box[:2]+box[2:])/2;half=(box[2:]-box[:2])*context_scale/2
    inside=valid&(original>=box[:2]).all(-1)&(original<box[2:]).all(-1)
    region=valid&(original>=center-half).all(-1)&(original<center+half).all(-1)
    return original,inside,region&~inside,valid


def greedy_2d(score,xy,inside,context,k=24,diversity=.1,context_cap=12):
    if k<1 or not torch.isfinite(score).all():raise ValueError('invalid sampler scores/budget')
    if len(score)!=len(xy) or len(score)!=len(inside) or len(score)!=len(context):raise ValueError('token layout mismatch')
    # Original flattened row-major patch index is the deterministic tie breaker.
    inside=inside.bool();context=context.bool()&~inside
    selected=[];distance=torch.full_like(score,float('inf'))
    available=inside|context
    target_inside=min(int(inside.sum()),(k+1)//2)
    for step in range(min(k,int(available.sum()))):
        mask=available.clone()
        if sum(bool(inside[i]) for i in selected)<target_inside:mask &= inside
        if sum(bool(context[i]) for i in selected)>=context_cap:mask &= ~context
        if not mask.any():break
        priority=score.clone()
        if selected:priority=priority+diversity*normalize_attention(distance)
        priority[~mask]=-torch.inf
        index=int(priority.argmax());selected.append(index);available[index]=False
        distance=torch.minimum(distance,torch.linalg.vector_norm(xy-xy[index],dim=-1))
    return torch.tensor(selected,device=score.device,dtype=torch.long)


class IdentityGuidedSampler(nn.Module):
    def __init__(self,token_dim,query_dim=128):
        super().__init__()
        self.query=nn.Linear(query_dim,query_dim,bias=False)
        self.key=nn.Linear(token_dim,query_dim,bias=False)
        self.mix=nn.Parameter(torch.zeros(3))

    def scores(self,tokens,attention,query,inside,mode):
        if mode not in ('A','B','C'):raise ValueError('unknown AG ablation')
        attn=normalize_attention(attention)
        related=F.cosine_similarity(self.key(tokens.float()),self.query(query)[None],dim=-1)
        weights=self.mix.softmax(0)
        if mode=='A':score=inside.float()
        elif mode=='B':score=attn+inside.float()
        else:score=weights[0]*attn+weights[1]*related+weights[2]*inside.float()
        return score,related

    def forward(self,tokens,attention,query,xy,inside,context,k=24,mode='C',soft=False):
        score,related=self.scores(tokens,attention,query,inside,mode)
        legal=inside.bool()|(context.bool() if mode!='A' else torch.zeros_like(context,dtype=torch.bool))
        if not legal.any():return tokens.new_empty((0,tokens.shape[-1])),torch.empty(0,device=tokens.device,dtype=torch.long),score
        if soft:
            # Differentiable warm-up pooling, then hard selection plus soft selected
            # weights in training. Random, untrained projections are never scored
            # as identity evidence in the reported learned AG-C experiments.
            weights=score.masked_fill(~legal,-torch.inf).softmax(0)
            return (weights[:,None]*tokens.float()).sum(0,keepdim=True),torch.where(legal)[0],score
        selected=greedy_2d(score,xy,inside,context&legal,k,diversity=0 if mode=='A' else .1,context_cap=0 if mode=='A' else k//2)
        return tokens[selected],selected,score
