"""Conditional calibration only on independently labeled candidate support."""
import torch
from .schemas import masked_softmax

def known_distribution(logits,known,positive,temperature=1.):
    probabilities=masked_softmax(logits/temperature,known)
    mass=(probabilities*positive).sum(-1)
    eligible=positive.any(-1)&known.any(-1)
    nll=-mass[eligible].clamp_min(1e-12).log()
    target=positive.float()/positive.sum(-1,keepdim=True).clamp_min(1)
    brier=((probabilities-target).square()*known).sum(-1)[eligible]
    return probabilities,nll,brier

def risk_mask(probabilities,legal,confidence=.55,margin=.10):
    if probabilities.shape[-1]==0:
        return torch.ones(probabilities.shape[:-1],dtype=torch.bool,device=probabilities.device)
    sortedp=probabilities.masked_fill(~legal,0).sort(-1,descending=True).values
    gap=sortedp[...,0]-sortedp[...,1] if sortedp.shape[-1]>1 else torch.ones_like(sortedp[...,0])
    return (sortedp[...,0]<confidence)|(gap<margin)|~legal.any(-1)
