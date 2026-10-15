"""GMT-only tiny student. No teacher, VGGT tokens or cache interface."""
import torch
from torch import nn


class TinyRelationStudent(nn.Module):
    def __init__(self,input_dim,max_residual=1.):
        super().__init__()
        if input_dim<1 or max_residual<0:raise ValueError('invalid student contract')
        self.input_dim=input_dim;self.max_residual=max_residual
        self.mlp=nn.Sequential(nn.Linear(input_dim,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
        nn.init.zeros_(self.mlp[-1].weight);nn.init.zeros_(self.mlp[-1].bias)

    def forward(self,evidence):
        if evidence.shape[-1]!=self.input_dim:raise ValueError('native evidence dimension mismatch')
        return self.mlp(evidence.float()).squeeze(-1).tanh()*self.max_residual
