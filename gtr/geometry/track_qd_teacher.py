"""Small task decoder; per-layer projections and track-dependent feature weights."""
import torch
from torch import nn
from .track_conditioned_see_query import TrackConditionedSeeQuery


class TrackQDTeacher(nn.Module):
    def __init__(self,channels,hidden=128,mode='track',pair_dim=8,projection_hidden=None):
        super().__init__()
        if len(channels)!=4 or mode not in ('last','mean','concat','global','track'):raise ValueError('invalid QD ablation')
        self.mode=mode
        active_channels=channels[-1:] if mode=='last' else channels
        self.projections=nn.ModuleList(
            nn.Sequential(nn.Linear(c,hidden),nn.LayerNorm(hidden)) if projection_hidden is None else
            nn.Sequential(nn.Linear(c,projection_hidden),nn.GELU(),nn.Linear(projection_hidden,hidden),nn.LayerNorm(hidden))
            for c in active_channels)
        self.see=TrackConditionedSeeQuery(hidden,conditioned=mode=='track') if mode in ('global','track') else None
        self.concat=nn.Linear(4*hidden,hidden) if mode=='concat' else None
        layer=nn.TransformerDecoderLayer(hidden,4,dim_feedforward=256,dropout=.1,batch_first=True)
        self.decoder=nn.TransformerDecoder(layer,2)
        self.score=nn.Sequential(nn.Linear(2*hidden+pair_dim,hidden),nn.GELU(),nn.Linear(hidden,1))

    def forward(self,query,layers,pair,valid=None):
        if len(layers)!=4 or any(len(x)!=len(layers[0]) for x in layers):raise ValueError('unaligned layer tokens')
        if not len(layers[0]):return query.sum()*0,None
        active_layers=layers[-1:] if self.mode=='last' else layers
        projected=torch.stack([proj(x.float()) for proj,x in zip(self.projections,active_layers)])
        if self.mode=='last':weights=query.new_tensor([0.,0.,0.,1.]);memory=projected[-1]
        elif self.mode=='mean':weights=query.new_full((4,),.25);memory=projected.mean(0)
        elif self.mode=='concat':weights=None;memory=self.concat(projected.permute(1,0,2).flatten(1))
        else:
            weights=self.see(query);memory=(projected*weights[:,None,None]).sum(0)
        if valid is not None:
            if valid.shape!=(len(memory),) or not valid.any():raise ValueError('empty/invalid QD padding mask')
            memory=memory[valid]
        if not torch.isfinite(memory).all():raise ValueError('nonfinite teacher evidence')
        context=self.decoder(query[None,None],memory[None])[0,0]
        return self.score(torch.cat((query,context,pair.float()))).squeeze(-1),weights
