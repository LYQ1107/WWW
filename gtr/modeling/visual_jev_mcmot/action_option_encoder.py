import torch
from torch import nn
from torch.nn import functional as F

class ActionOptionEncoder(nn.Module):
    def __init__(self,d=128):
        super().__init__()
        self.visual_projection = nn.Linear(1152,d)
        self.history_query = nn.Linear(d,d)
        self.evidence_projection = nn.Sequential(nn.Linear(12,d),nn.GELU(),nn.Linear(d,d))
        self.semantic_descriptor = nn.Embedding(6,d)
        self.norm = nn.LayerNorm(d)

    def forward(self,question,options,evidence):
        visual = self.visual_projection(F.normalize(options.visual,dim=-1,eps=1e-8))
        similarity = (visual*self.history_query(question)[:,:,None,None,:]).sum(-1)/(visual.shape[-1]**.5)
        mask = options.visual_mask
        from .schemas import masked_softmax
        weights = masked_softmax(similarity,mask)
        history = (weights[...,None]*visual).sum(-2)
        return self.norm(history + self.evidence_projection(evidence) + self.semantic_descriptor(options.kinds))
