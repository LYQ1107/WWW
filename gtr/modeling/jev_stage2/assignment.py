"""Each detection has a private terminal; per-camera identity capacity is one."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

def lawful_choice(logits,legal):
    d,k=legal.shape
    if not d:return []
    assert logits.shape==(d,k+1) and torch.isfinite(logits).all()
    cost=np.full((d,k+d),1e9,dtype=np.float64);values=logits.detach().cpu().double().numpy();mask=legal.detach().cpu().numpy()
    cost[:,:k]=np.where(mask,-values[:,:k],1e9);cost[np.arange(d),k+np.arange(d)]=-values[:,-1]
    rows,cols=linear_sum_assignment(cost);result=[-1]*d
    for row,col in zip(rows,cols):
        if col<k:assert mask[row,col];result[row]=int(col)
    return result

def structured_assignment_loss(logits,legal,targets,known):
    terms=[]
    for b in range(len(logits)):
        rows=torch.where(known[b])[0]
        if not len(rows):continue
        gt=targets[b,rows];k=legal.shape[-1];existing=gt[gt<k]
        if len(existing)!=len(existing.unique()):continue
        score=logits[b,rows];mask=legal[b,rows];augment=torch.ones_like(score);augment[torch.arange(len(rows),device=score.device),gt]=0
        selected=lawful_choice(score+augment,mask);chosen=torch.tensor([k if c<0 else c for c in selected],device=score.device);margin=(chosen!=gt).float().sum();terms.append(torch.relu(score[torch.arange(len(rows)),chosen].sum()-score[torch.arange(len(rows)),gt].sum()+margin)/len(rows))
    return torch.stack(terms).mean() if terms else logits.sum()*0
