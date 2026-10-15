"""Independent Phase2 arms vectorized across model slices, not across queries."""
import torch
from torch import nn
import torch.nn.functional as F
from pirga_innovation import InnovationHead,ARMS


class GroupedInnovations(nn.Module):
    def __init__(self,seed=20261015):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);head=InnovationHead(ARMS[0])
        for name,param in head.named_parameters():
            self.register_parameter(name.replace('.','__'),nn.Parameter(torch.stack([param.detach().clone() for _ in ARMS])))

    def arm_state(self,index):
        return {name.replace('__','.'):v[index].detach().cpu().clone() for name,v in self.named_parameters()}

    def forward(self,state,query,native,no_innovation=False):
        b=len(ARMS);conditions=state.conditions.unsqueeze(0).expand(b,-1,-1).clone()
        conditions[ARMS.index('A6_no_camera'),:,:3]=0
        conditions[ARMS.index('A7_no_time'),:,3:5]=0;conditions[ARMS.index('A7_no_time'),:,7:]=0
        x=torch.cat((state.mean.unsqueeze(0).expand(b,-1,-1),conditions),dim=-1)
        latent=F.gelu(torch.bmm(x,self.down__weight.transpose(1,2))+self.down__bias[:,None,:])
        residual=.25*torch.tanh(torch.bmm(latent,self.up__weight.transpose(1,2))+self.up__bias[:,None,:])
        mu=F.normalize(state.mean[None]+residual,dim=-1)
        import math
        low,high=math.log(.05),math.log(4.)
        lv=(low+high)/2+(high-low)/2*torch.tanh((torch.bmm(latent,self.sigma__weight.transpose(1,2))+self.sigma__bias[:,None,:]).squeeze(-1))
        base=native+2*(state.mean@query)
        innovation=((mu@query)-(F.normalize(state.mean,dim=-1)@query)[None]).clamp(-.25,.25)
        gate=torch.stack([torch.exp(-lv[i].clamp_min(0)) if arm in ('A4_uncertainty','A6_no_camera','A7_no_time','AS_same_camera') else torch.ones_like(lv[i]) for i,arm in enumerate(ARMS)])
        score=base[None]+2*(state.counts/(state.counts+1))[None]*gate*innovation
        if no_innovation:score=base[None].expand(b,-1)
        return score,mu,lv

    def clip_per_arm(self,maximum=1.):
        norm=torch.sqrt(sum((p.grad.flatten(1)**2).sum(1) for p in self.parameters() if p.grad is not None))
        multiplier=(maximum/(norm+1e-6)).clamp(max=1)
        for p in self.parameters():
            if p.grad is not None:p.grad.mul_(multiplier.view((len(ARMS),)+(1,)*(p.ndim-1)))
        return norm


def grouped_innovation_losses(scores,mu,lv,state,query,label):
    from pirga_innovation import innovation_losses
    values=[innovation_losses(scores[i],mu[i],lv[i],state,query,label,arm) for i,arm in enumerate(ARMS)]
    return tuple(torch.stack([v[i] for v in values]) for i in range(3)),[v[3] for v in values]
