"""Small history-only identity-state predictor and explicitly bounded innovation.

The predictor sees no query. The query appears only in scoring/offline losses.
Ordinary, contrastive and predictive arms allocate identical parameter tensors.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from pirga_candidate_state import CandidateState

ARMS=('A2_rank','AC_contrastive','A4_uncertainty','A5_no_uncertainty','A6_no_camera','A7_no_time','AS_same_camera')


class InnovationHead(nn.Module):
    def __init__(self,arm):
        super().__init__()
        if arm not in ARMS:raise ValueError('Unregistered innovation arm')
        self.arm=arm;self.down=nn.Linear(1163,16);self.up=nn.Linear(16,1152);self.sigma=nn.Linear(16,1)
        nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias);nn.init.zeros_(self.sigma.weight)
        low,high=math.log(.05),math.log(4.)
        nn.init.constant_(self.sigma.bias,math.atanh(-(low+high)/(high-low)))

    def predict(self,state):
        if not isinstance(state,CandidateState):raise TypeError('Only GT-free CandidateState is accepted')
        conditions=state.conditions.clone()
        if self.arm=='A6_no_camera':conditions[:,:3]=0
        if self.arm=='A7_no_time':conditions[:,3:5]=0;conditions[:,7:]=0
        latent=F.gelu(self.down(torch.cat((state.mean,conditions),dim=-1)))
        residual=.25*torch.tanh(self.up(latent));mu=F.normalize(state.mean+residual,dim=-1)
        low,high=math.log(.05),math.log(4.)
        logvar=(low+high)/2+(high-low)/2*torch.tanh(self.sigma(latent).squeeze(-1))
        return mu,logvar

    def forward(self,state,query,native,no_innovation=False):
        if query.shape!=(1152,) or native.shape!=(len(state.mean),):raise ValueError('Full query/candidate dimensions required')
        base=native+2*(state.mean@query)
        mu,logvar=self.predict(state)
        if no_innovation:return base,mu,logvar
        # Both terms use the same cosine normalization. Their exact equality
        # makes zero residual exactly C1, including floating-point arithmetic.
        innovation=(mu@query-F.normalize(state.mean,dim=-1)@query).clamp(-.25,.25)
        support=state.counts/(state.counts+1)
        gate=torch.exp(-logvar.clamp_min(0)) if self.arm in ('A4_uncertainty','A6_no_camera','A7_no_time','AS_same_camera') else torch.ones_like(logvar)
        return base+2*support*gate*innovation,mu,logvar


def innovation_losses(scores,mu,logvar,state,query,label,arm):
    """TRAIN-only certification masks, same candidates; no UNKNOWN negatives."""
    positive=torch.tensor(label['certified_target_indices'],dtype=torch.long,device=scores.device)
    known=torch.tensor(label['candidate_known_mask'],dtype=torch.bool,device=scores.device)
    if label['GT_person'] is None or not len(positive) or not known[positive].all():raise ValueError('No legal TRAIN supervision')
    ranking=torch.logsumexp(scores[known],dim=0)-torch.logsumexp(scores[positive],dim=0)
    cross=state.cross_counts[positive]>0
    valid=(state.counts[positive]-state.cross_counts[positive]>0) if arm=='AS_same_camera' else cross
    target=positive[valid];zero=scores.sum()*0
    if arm in ('A4_uncertainty','A6_no_camera','A7_no_time','AS_same_camera') and len(target):
        error=1-(mu[target]*query.detach()).sum(-1)
        prediction=(error*torch.exp(-logvar[target])+logvar[target]).mean()
    elif arm=='A5_no_uncertainty' and len(target):prediction=(1-(mu[target]*query.detach()).sum(-1)).mean()
    elif arm=='AC_contrastive':
        cosine=mu@query.detach()/.1
        prediction=torch.logsumexp(cosine[known],dim=0)-torch.logsumexp(cosine[positive],dim=0)
    else:prediction=zero
    return ranking+.5*prediction,ranking,prediction,len(target)
