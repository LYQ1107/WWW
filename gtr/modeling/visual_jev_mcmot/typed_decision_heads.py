import torch
from torch import nn
from .schemas import masked_softmax

class TypedDecisionHeads(nn.Module):
    def __init__(self,d=128):
        super().__init__()
        self.choice = nn.Linear(d,1)
        self.binary = nn.Linear(d,1)
        self.abstain = nn.Linear(d,1)
        self.register_buffer('temperatures',torch.ones(3))

    def forward(self,representations,question,types,kinds,legal):
        choice = self.choice(representations).squeeze(-1)
        binary = self.binary(representations).squeeze(-1)
        logits = torch.where(types[...,None] == 2,binary,choice)
        logits = logits/self.temperatures[types][...,None].clamp_min(.05)
        probabilities = masked_softmax(logits,legal)
        write = (probabilities*((kinds == 4)&legal)).sum(-1)
        return {'choice_logits':logits,'probabilities':probabilities,'write_probability':write,'abstain_logit':self.abstain(question).squeeze(-1),'abstain_calibrated':False}
