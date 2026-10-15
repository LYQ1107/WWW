"""Model-wise vectorization of three independent legacy heads, query batch1.

No cross-arm learning or averaging: each arm has its own parameter slice and
AdamW state. Total loss is a sum and gradient clipping is per arm. Extracted
weights retain the original CandidatePredictor state_dict format.
"""
import torch
from torch import nn
import torch.nn.functional as F
from cv_pil_v2_model import CandidatePredictor

NAMES=('down.weight','down.bias','up.weight','up.bias')


class GroupedHeads(nn.Module):
    def __init__(self,states=None,seed=20261015):
        super().__init__()
        if states is None:
            torch.manual_seed(seed);state=CandidatePredictor().state_dict();states=[state,state,state]
        self.down_weight=nn.Parameter(torch.stack([s['down.weight'].detach().clone() for s in states]))
        self.down_bias=nn.Parameter(torch.stack([s['down.bias'].detach().clone() for s in states]))
        self.up_weight=nn.Parameter(torch.stack([s['up.weight'].detach().clone() for s in states]))
        self.up_bias=nn.Parameter(torch.stack([s['up.bias'].detach().clone() for s in states]))

    def forward(self,b):
        h=b.histories;cond=h.new_zeros((3,len(h),3))
        cond[:2,:,b.destination_camera-1]=1.;cond[:,:,2]=b.age
        x=torch.cat((h.unsqueeze(0).expand(3,-1,-1),cond),dim=-1)
        z=F.gelu(torch.bmm(x,self.down_weight.transpose(1,2))+self.down_bias[:,None,:])
        residual=torch.bmm(z,self.up_weight.transpose(1,2))+self.up_bias[:,None,:]
        predicted=F.normalize(h[None,:,:]+.25*torch.tanh(residual),dim=-1)
        n=len(b.native_scores);owner=(b.owners[None,:]+torch.arange(3,device=h.device)[:,None]*n).reshape(-1)
        proto=h.new_zeros((3*n,1152)).index_add(0,owner,predicted.reshape(-1,1152)).reshape(3,n,1152)
        scores=b.native_scores[None,:]+2*(F.normalize(proto,dim=-1)@b.query)
        return scores,predicted

    def arm_state(self,arm):
        return {name:p[arm].detach().cpu().clone() for name,p in zip(NAMES,(self.down_weight,self.down_bias,self.up_weight,self.up_bias))}

    def clip_per_arm(self,max_norm=1.):
        norms=torch.stack([torch.linalg.vector_norm(p.grad.flatten(1),ord=2,dim=1) for p in self.parameters()],dim=1)
        total=torch.linalg.vector_norm(norms,ord=2,dim=1);coef=(max_norm/(total+1e-6)).clamp(max=1.)
        for p in self.parameters():p.grad.mul_(coef.reshape([3]+[1]*(p.ndim-1)))
        return total


def grouped_losses(scores,predicted,b,label):
    positive=torch.tensor(label['certified_target_indices'],dtype=torch.long,device=scores.device)
    known=torch.tensor(label['candidate_known_mask'],dtype=torch.bool,device=scores.device)
    if not len(positive) or not known[positive].all():raise ValueError('Uncertified targets')
    rank=torch.logsumexp(scores[:,known],dim=1)-torch.logsumexp(scores[:,positive],dim=1)
    mask=b.cross_view & torch.isin(b.owners,positive)
    pred=(1-(predicted[:,mask,:]*b.query.detach()).sum(-1)).mean(-1) if mask.any() else scores.sum(-1)*0.
    loss=rank+pred*scores.new_tensor([0.,.5,.5])
    return loss,rank,pred,int(mask.sum())
