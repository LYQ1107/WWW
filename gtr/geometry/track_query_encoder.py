"""Queries from observed GMT history only; no identity label inputs."""
import torch
from torch import nn
from torch.nn import functional as F


class TrackQueryEncoder(nn.Module):
    def __init__(self,input_dim,hidden=128,summary='mean'):
        super().__init__()
        if input_dim<=0 or summary not in ('mean','ema'): raise ValueError('invalid query contract')
        self.input_dim,self.summary = input_dim,summary
        self.mlp = nn.Sequential(nn.Linear(2*input_dim+4,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.LayerNorm(hidden))

    def forward(self,history,history_times,history_cameras,current_time,current_camera,observed_length=None):
        if history.ndim!=2 or history.shape[1]!=self.input_dim or len(history)==0:
            raise ValueError('query requires actual nonempty GMT history with observed feature dimension')
        if len(history_times)!=len(history) or len(history_cameras)!=len(history):raise ValueError('history metadata mismatch')
        if not torch.isfinite(history).all() or not torch.isfinite(history_times).all():raise ValueError('invalid history')
        if bool((history_times>current_time).any()):raise ValueError('future observation in track query')
        order=torch.argsort(history_times,stable=True)[-4:]
        values=F.normalize(history[order].float(),dim=-1)
        recent=values[-1]
        if self.summary=='mean':long=values.mean(0)
        else:
            weights=values.new_tensor([.5**i for i in reversed(range(len(values)))])
            long=(values*weights[:,None]).sum(0)/weights.sum()
        delta=(current_time-history_times[order[-1]]).float()
        # Equality, not camera ordinal, supports arbitrary physical camera IDs.
        same=(history_cameras[order[-1]]==current_camera).float()
        length=len(history) if observed_length is None else observed_length
        temporal=torch.stack((torch.log1p(delta),same,1-same,delta.new_tensor(float(length)).log1p()))
        return self.mlp(torch.cat((recent,F.normalize(long,dim=0),temporal)))
