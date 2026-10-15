"""Visible causal A(previous)->B(current)->A(current), with absorbing NULL."""
import torch


def temporal_cycle_loss(p_ab,p_ba,p_aa,visible_rows,previous_time,current_time):
    if previous_time>=current_time:raise ValueError('cycle requires a strictly past first observation')
    if p_ab.ndim!=2 or p_ba.ndim!=2 or p_aa.ndim!=2:raise ValueError('cycle matrices must be two-dimensional')
    a,b=p_ab.shape[0],p_ba.shape[0];c=p_aa.shape[1]
    if p_ab.shape[1]!=b+1 or p_ba.shape[1]!=c or p_aa.shape[0]!=a or visible_rows.shape!=(a,):raise ValueError('cycle identity-space/NULL layout mismatch')
    if any(not torch.isfinite(p).all() or bool((p<0).any()) for p in (p_ab,p_ba,p_aa)):raise ValueError('invalid relation probabilities')
    for p in (p_ab,p_ba,p_aa):
        if len(p) and not torch.allclose(p.sum(-1),p.new_ones(len(p)),atol=1e-5):raise ValueError('rows must include a normalized NULL state')
    if not b or not visible_rows.any():return (p_ab.sum()+p_ba.sum()+p_aa.sum())*0
    null=p_ba.new_zeros((1,c));null[0,-1]=1
    indirect=p_ab@torch.cat((p_ba,null),0)
    direct=p_aa
    left=(direct.detach()*(direct.detach().clamp_min(1e-8).log()-indirect.clamp_min(1e-8).log())).sum(-1)
    right=(indirect.detach()*(indirect.detach().clamp_min(1e-8).log()-direct.clamp_min(1e-8).log())).sum(-1)
    return ((left+right)*.5)[visible_rows].mean()
