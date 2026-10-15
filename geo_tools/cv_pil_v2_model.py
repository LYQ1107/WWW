"""CV-PIL-owned tiny heads. Actor inputs never contain GT or future features."""
from dataclasses import dataclass
import hashlib

import torch
from torch import nn
import torch.nn.functional as F

from cv_pil_native_handoff import INPUT_KEYS


@dataclass(frozen=True)
class ActorBatch:
    query: torch.Tensor
    histories: torch.Tensor
    owners: torch.Tensor
    age: torch.Tensor
    cross_view: torch.Tensor
    native_scores: torch.Tensor
    destination_camera: int


def tensorize(actor: dict, age_denominator=400.) -> ActorBatch:
    if set(actor) != INPUT_KEYS:
        raise ValueError('Only the exact GT-free actor schema is accepted')
    if isinstance(actor['camera'],bool) or actor['camera'] not in (1,2):
        raise ValueError('Invalid physical destination camera')
    q=actor['query_feature'].detach().float().clone()
    native=actor['native_scores'].detach().float().clone()
    n=len(actor['native_candidate_ids'])
    if q.shape!=(1152,) or native.shape!=(n,) or len(actor['past_histories'])!=n:
        raise ValueError('Actor feature/candidate dimensions differ')
    if not torch.isfinite(q).all() or q.norm()==0 or not torch.isfinite(native).all():
        raise ValueError('Invalid actor features or native scores')
    histories=[];owners=[];age=[];cross=[]
    for i,rows in enumerate(actor['past_histories']):
        for r in rows:
            if set(r)!={'camera','lag','feature','cross_view'} or r['lag']<=0:
                raise ValueError('Illegal or labeled actor history')
            if r['camera'] not in (1,2) or r['cross_view']!=(r['camera']!=actor['camera']):
                raise ValueError('Physical cross-view metadata differs')
            x=r['feature'].detach().float().clone()
            if x.shape!=(1152,) or not torch.isfinite(x).all() or x.norm()==0:
                raise ValueError('Bad full native history vector')
            histories.append(x);owners.append(i);age.append(min(1.,r['lag']/age_denominator));cross.append(r['cross_view'])
    h=torch.stack(histories) if histories else torch.zeros(0,1152)
    return ActorBatch(F.normalize(q,dim=0),F.normalize(h,dim=-1),
                      torch.tensor(owners,dtype=torch.long),torch.tensor(age,dtype=torch.float32),
                      torch.tensor(cross,dtype=torch.bool),native,int(actor['camera']))


def fused_scores(batch: ActorBatch, projected: torch.Tensor, alpha=2.):
    n=len(batch.native_scores)
    prototypes=torch.zeros(n,1152,dtype=projected.dtype,device=projected.device)
    prototypes=prototypes.index_add(0,batch.owners,projected)
    # Cosine normalization makes sum equivalent to mean; empty histories stay zero.
    return batch.native_scores + alpha*(F.normalize(prototypes,dim=-1)@batch.query)


class CandidatePredictor(nn.Module):
    """Same trainable tensors for C2/C3/C4; C4 zeros explicit camera channels."""
    def __init__(self,use_camera=True):
        super().__init__();self.use_camera=use_camera
        self.down=nn.Linear(1152+3,16)
        self.up=nn.Linear(16,1152)
        nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)

    def project(self,batch):
        h=batch.histories
        cond=torch.zeros(len(h),3,dtype=h.dtype,device=h.device)
        if self.use_camera:cond[:,batch.destination_camera-1]=1.
        cond[:,2]=batch.age
        return F.normalize(h+.25*torch.tanh(self.up(F.gelu(self.down(torch.cat((h,cond),dim=-1))))),dim=-1)

    def forward(self,batch: ActorBatch):
        if not isinstance(batch,ActorBatch):raise TypeError('ActorBatch required, no labeled event dicts')
        predicted=self.project(batch)
        return fused_scores(batch,predicted),predicted


def training_eligible(batch,supervision):
    positives=set(supervision['certified_target_indices'])
    return bool(positives and len(batch.histories) and any(
        known and i not in positives for i,known in enumerate(supervision['candidate_known_mask'])))


def supervised_losses(scores,predicted,batch,supervision):
    """OFFLINE TRAIN loss only. UNKNOWN never becomes a certified negative."""
    positives=torch.tensor(supervision['certified_target_indices'],dtype=torch.long)
    known=torch.tensor(supervision['candidate_known_mask'],dtype=torch.bool)
    if not len(positives) or not known[positives].all():raise ValueError('Uncertified ranking targets')
    ranking=torch.logsumexp(scores[known],dim=0)-torch.logsumexp(scores[positives],dim=0)
    mask=batch.cross_view & torch.isin(batch.owners,positives)
    # Target is later than source, but is only an offline TRAIN objective. The
    # source predictor consumes history, destination camera and elapsed time.
    prediction=(1-(predicted[mask]*batch.query.detach()).sum(-1)).mean() if mask.any() else scores.sum()*0.
    return ranking,prediction,int(mask.sum())


def parameter_digest(model):
    h=hashlib.sha256()
    for name,x in model.state_dict().items():
        h.update(name.encode());h.update(x.detach().cpu().numpy().tobytes())
    return h.hexdigest()
