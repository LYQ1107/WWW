"""Shared low-rank target projection; exact invariance to the two order slots."""
import torch
from torch import nn
from torch.nn import functional as F

class LowRankIdentityHead(nn.Module):
    def __init__(self,gmt_dim,foundation_dim=0,hidden_dim=96,output_dim=32,symmetric=False):
        super().__init__()
        self.gmt_dim=gmt_dim;self.foundation_dim=foundation_dim;self.output_dim=output_dim;self.symmetric=symmetric
        if symmetric and not foundation_dim:raise ValueError('symmetric projection requires foundation features')
        self.encoder=nn.Sequential(nn.Linear(gmt_dim+foundation_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,output_dim))
        # Initial head is exactly neutral; no detector/transformer parameters.
        self.scale=nn.Parameter(torch.zeros(()));self.bias=nn.Parameter(torch.zeros(()))

    def encode(self,gmt,foundation=None):
        if gmt.shape[-1]!=self.gmt_dim:raise ValueError('GMT feature dimension mismatch')
        gmt=F.normalize(gmt.float(),dim=-1)
        if self.foundation_dim:
            count=2 if self.symmetric else 1
            if foundation is None or foundation.shape[-1]!=count*self.foundation_dim:raise ValueError('foundation feature dimension mismatch')
            # Apply the same learned projection to original and reversed camera
            # slots separately. Swapping the slots only permutes output slots.
            outputs=[F.normalize(self.encoder(torch.cat([gmt,F.normalize(chunk.float(),dim=-1)],dim=-1)),dim=-1)
                     for chunk in foundation.split(self.foundation_dim,dim=-1)]
            return torch.cat(outputs,dim=-1)/count**.5
        return F.normalize(self.encoder(gmt),dim=-1)

    def evidence(self,query,reference):
        return torch.tanh(self.scale*(query@reference.T)+self.bias)

    def forward(self,gmt_query,gmt_reference,foundation_query=None,foundation_reference=None):
        return self.evidence(self.encode(gmt_query,foundation_query),self.encode(gmt_reference,foundation_reference))

def matched_hidden_width(gmt_dim,foundation_dim,target_parameters,output_dim=32):
    # Parameters = hidden*(input + output + 1) + output bias + scale/bias.
    return max(1,(target_parameters-output_dim-2)//(gmt_dim+foundation_dim+output_dim+1))
