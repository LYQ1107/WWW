"""Bounded unique-track residual transport into unchanged proposal logits."""
import torch


class IdentityRelationAdapter:
    def __init__(self,enabled=True,max_residual=1.):
        if max_residual<0:raise ValueError('negative residual bound')
        self.enabled,self.max_residual=enabled,max_residual

    def __call__(self,logits,proposal_tracks,candidate_tracks,track_residual,legal=None):
        if not self.enabled or self.max_residual==0:return logits
        if logits.ndim!=2 or len(proposal_tracks)!=logits.shape[1]:raise ValueError('proposal layout mismatch')
        if candidate_tracks.shape!=track_residual.shape or candidate_tracks.shape[0]!=logits.shape[0]:raise ValueError('candidate layout mismatch')
        if not logits.numel() or not candidate_tracks.numel():return logits
        if not torch.isfinite(logits).all():return logits
        if legal is None:legal=torch.ones_like(candidate_tracks,dtype=torch.bool)
        if legal.shape!=candidate_tracks.shape:raise ValueError('legal mask mismatch')
        valid=legal&torch.isfinite(track_residual)&(candidate_tracks>=0)
        for row in range(len(logits)):
            identities=candidate_tracks[row][valid[row]]
            if identities.unique().numel()!=len(identities):raise ValueError('duplicate candidate track; aggregate before transport')
        bounded=torch.where(valid,track_residual.clamp(-self.max_residual,self.max_residual),torch.zeros_like(track_residual))
        if not bool((bounded!=0).any()):return logits
        hit=(candidate_tracks[:,:,None]==proposal_tracks[None,None,:])&valid[:,:,None]
        counts=hit.sum(-1).clamp_min(1)
        # Sum of transported logit residuals for a unique track equals its one
        # bounded residual, independent of proposal count. Original per-frame
        # Softmax/GTA/threshold remain nonlinear and are never replaced.
        delta=(hit.to(logits.dtype)*(bounded/counts)[:,:,None]).sum(1)
        return logits+delta.to(logits.dtype)
