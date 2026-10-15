"""Frozen nonlearned history aggregation controls; no query or GT inputs."""
import torch
import torch.nn.functional as F

METHODS=('R_mean','R_top4_recent','R_coordinate_median','R_trim10','R_latest','R_decay')


def prototypes(batch,raw_lags,tau):
    if tau<=0 or len(raw_lags)!=len(batch.histories) or (len(raw_lags) and (raw_lags<=0).any()):raise ValueError('Noncausal history or invalid fixed decay')
    n=len(batch.native_scores);h=batch.histories;owners=batch.owners
    sums=torch.zeros(n,1152).index_add_(0,owners,h)
    result={'R_mean':F.normalize(sums,dim=-1)}
    others={m:torch.zeros(n,1152) for m in METHODS if m!='R_mean'}
    for i in range(n):
        mask=owners==i;x=h[mask];ages=raw_lags[mask]
        if not len(x):continue
        order=torch.argsort(ages,stable=True);recent=x[order[:4]]
        others['R_top4_recent'][i]=recent.sum(0)
        others['R_latest'][i]=x[order[0]]
        others['R_coordinate_median'][i]=x.median(0).values
        trimmed=int(len(x)*.1);sorted_x=x.sort(0).values
        others['R_trim10'][i]=sorted_x[trimmed:len(x)-trimmed].sum(0)
        weights=torch.exp(-(ages-ages.min())/tau)
        others['R_decay'][i]=(x*weights[:,None]).sum(0)
    result.update({k:F.normalize(v,dim=-1) for k,v in others.items()})
    return result
