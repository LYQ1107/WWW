import torch
from torch import nn
from .visual_state_encoder import ReaderBlock

class ActionOptionReader(nn.Module):
    def __init__(self,d=128,heads=4):
        super().__init__()
        self.question_reader = ReaderBlock(d,heads)
        self.state_reader = ReaderBlock(d,heads)
        self.competition = nn.Linear(2*d,d)
        self.question_projection = nn.Linear(d,d)
        self.option_projection = nn.Linear(d,d)
        self.norm = nn.LayerNorm(d)
        self.output_norm = nn.LayerNorm(d)

    def forward(self,question,memory,memory_mask,options,legal,gating=True,read=True):
        b,q,k,d = options.shape
        if k == 0 or q == 0:
            return options
        x = options.reshape(b,q*k,d)
        if read:
            # Each candidate sees only its own dynamic question.
            xx = x.reshape(b*q,k,d)
            qm = torch.ones((b*q,1),device=x.device,dtype=torch.bool)
            xx = self.question_reader(xx,question.reshape(b*q,1,d),qm)
            x = self.state_reader(xx.reshape(b,q*k,d),memory,memory_mask)
        x = x.reshape(b,q,k,d)
        count = legal.sum(-1,keepdim=True).clamp_min(1)
        mean = (x*legal[...,None]).sum(-2)/count
        maximum = x.masked_fill(~legal[...,None],-torch.inf).amax(-2)
        maximum = torch.where(torch.isfinite(maximum),maximum,torch.zeros_like(maximum))
        x = x + self.competition(torch.cat((mean,maximum),-1))[:,:,None,:]
        if gating:
            gate = torch.tanh(self.question_projection(question))[:,:,None,:]
            x = x + self.option_projection(self.norm(x))*gate
        return self.output_norm(x)*legal[...,None]
