"""Strong ordinary pairwise MLP; only its training objective differs D1/D2."""
import torch
from torch import nn
import torch.nn.functional as F

ARMS=('D1_MLP','D2_MLP_native_margin')


def decision_input(state,query,native,thresholds,opp,original_accepted):
    n=len(native);old=torch.zeros(n)
    if original_accepted>=0:old[original_accepted]=1
    context=torch.stack([native/(1+abs(native)),thresholds/(1+abs(thresholds)),opp/(1+abs(opp)),old],dim=-1)
    return torch.cat([query[None].expand(n,-1),state.mean,state.mean*query,state.conditions,context],dim=-1)


class DecisionMLP(nn.Module):
    def __init__(self):
        super().__init__();self.down=nn.Linear(3471,11);self.up=nn.Linear(11,1);nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
    def forward(self,x,state,query,native):
        if x.shape!=(len(native),3471):raise ValueError('Pair/native information contract changed')
        delta=.5*(state.counts/(state.counts+1))*torch.tanh(self.up(F.gelu(self.down(x))).squeeze(-1))
        return native+2*(state.mean@query)+delta,delta


def decision_losses(scores,delta,label,thresholds,opp,original_accepted,margin_on):
    if label['GT_person'] is None:raise ValueError('UNKNOWN cannot supervise identity decisions')
    pos=torch.tensor(label['certified_target_indices'],dtype=torch.long);known=torch.tensor(label['candidate_known_mask'],dtype=torch.bool)
    if not len(pos) or not known[pos].all():raise ValueError('No certified legal positive')
    negative=known.clone();negative[pos]=False
    rank=torch.logsumexp(scores[known],dim=0)-torch.logsumexp(scores[pos],dim=0)
    acceptance=F.relu(.1-(scores[pos]-thresholds[pos]).max())
    global_scores=scores+opp
    competition=F.relu(.1+torch.logsumexp(global_scores[negative],dim=0)-torch.logsumexp(global_scores[pos],dim=0)) if negative.any() else scores.sum()*0
    preserve=delta[original_accepted].square() if label['actual_primary_state']=='correct_link' and original_accepted in pos.tolist() else scores.sum()*0
    margin=acceptance+competition
    total=rank+.5*margin+.1*preserve if margin_on else rank
    return total,rank,margin,preserve


class GroupedDecisions(nn.Module):
    def __init__(self,seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):torch.manual_seed(seed);m=DecisionMLP()
        for name,v in m.named_parameters():self.register_parameter(name.replace('.','__'),nn.Parameter(torch.stack([v.detach().clone()]*2)))
    def forward(self,x,state,query,native):
        h=F.gelu(torch.matmul(x[None],self.down__weight.transpose(1,2))+self.down__bias[:,None,:])
        delta=.5*(state.counts/(state.counts+1))[None]*torch.tanh((torch.bmm(h,self.up__weight.transpose(1,2))+self.up__bias[:,None,:]).squeeze(-1))
        return (native+2*(state.mean@query))[None]+delta,delta
    def clip_per_arm(self):
        norm=torch.sqrt(sum(p.grad.flatten(1).square().sum(1) for p in self.parameters() if p.grad is not None));mult=(1/(norm+1e-6)).clamp(max=1)
        for p in self.parameters():
            if p.grad is not None:p.grad.mul_(mult.view((2,)+(1,)*(p.ndim-1)))
        return norm
    def arm_state(self,i):return {name.replace('__','.'):v[i].detach().clone() for name,v in self.named_parameters()}
