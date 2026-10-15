"""Global See-Query control and per-history-track four-layer conditioning."""
import torch
from torch import nn


class TrackConditionedSeeQuery(nn.Module):
    def __init__(self,hidden=128,conditioned=True):
        super().__init__();self.conditioned=conditioned
        self.base=nn.Parameter(torch.zeros(hidden))
        self.condition=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.norm=nn.LayerNorm(hidden)
        self.task_head=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,4))

    def forward(self,query):
        token=self.base+(self.condition(query) if self.conditioned else torch.zeros_like(query))
        return self.task_head(self.norm(token)).softmax(-1)
