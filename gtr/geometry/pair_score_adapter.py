"""Bounded residuals in proposal-logit space; zero/OFF return original objects."""
import torch

class PairScoreAdapter:
    def __init__(self, enabled=False, feature_weight=0., geometry_weight=0., max_residual=1.):
        self.enabled=enabled;self.feature_weight=feature_weight;self.geometry_weight=geometry_weight;self.max_residual=max_residual

    def apply(self,logits,feature_evidence=None,geometry_evidence=None,legal_mask=None):
        if not self.enabled or (self.feature_weight==0 and self.geometry_weight==0):return logits
        delta=torch.zeros_like(logits)
        for weight,value in [(self.feature_weight,feature_evidence),(self.geometry_weight,geometry_evidence)]:
            if weight and value is not None:
                if value.shape!=logits.shape:raise ValueError('candidate evidence shape mismatch')
                delta=delta+weight*torch.where(torch.isfinite(value),value,torch.zeros_like(value))
        if legal_mask is not None:delta=delta*legal_mask
        delta=delta.clamp(-self.max_residual,self.max_residual)
        return logits+delta

def calibrated_cosine(v1,v2,slope=1.,intercept=0.,max_evidence=1.):
    """Scalar calibrator parameters must be fitted on association TRAIN, never TEST."""
    raw=v2.float()@v1.float().T
    return torch.tanh((slope*raw+intercept)/max_evidence)*max_evidence

def normalize_candidate_scores(scores,legal_mask,std_floor=1e-3):
    """Remove row-common appearance bias before calibrating bounded logit evidence.

Only legal, finite candidates contribute. Empty/singleton/flat rows are neutral.
    """
    mask=legal_mask&torch.isfinite(scores)
    values=torch.where(mask,scores,torch.zeros_like(scores));count=mask.sum(1,keepdim=True)
    mean=values.sum(1,keepdim=True)/count.clamp_min(1)
    centered=torch.where(mask,scores-mean,torch.zeros_like(scores))
    variance=(centered**2).sum(1,keepdim=True)/count.clamp_min(1)
    usable=mask&(count>=2)&(variance>1e-12)
    normalized=centered/variance.sqrt().clamp_min(std_floor)
    return torch.where(usable,normalized,torch.zeros_like(normalized)),usable
