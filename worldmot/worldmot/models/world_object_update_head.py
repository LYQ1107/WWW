"""V3 history-conditioned identity assignment, lifecycle and memory correction."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

MODEL_SCHEMA = "worldobject_identity_assignment_joint_birth_v3_1"


class WorldObjectUpdateHead(nn.Module):
    """D comes from frozen GMT manifest, not from a synthetic bbox descriptor.

    Inputs: history[M,D], observations[N,D], geometry[M,N,12],
    observation_context[N,4], history_context[M,4]. No GT identity is an input.
    """
    def __init__(self, visual_dim, hidden_dim=128, geometry_dim=16,birth_pair_chunk=4096):
        super().__init__()
        self.visual_dim, self.hidden_dim, self.geometry_dim = int(visual_dim), int(hidden_dim), int(geometry_dim)
        self.roi_projection = nn.Linear(self.visual_dim, hidden_dim)
        self.pair_encoder = nn.Sequential(nn.Linear(3*hidden_dim + geometry_dim + 8, hidden_dim),
                                          nn.LayerNorm(hidden_dim), nn.GELU(),
                                          nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.assignment = nn.Linear(hidden_dim, 1)
        self.update = nn.Linear(hidden_dim, 1)
        self.memory = nn.Linear(hidden_dim, self.visual_dim)
        self.birth = nn.Sequential(nn.Linear(hidden_dim+8, hidden_dim), nn.GELU(), nn.Linear(hidden_dim,1))
        self.lifecycle = nn.Sequential(nn.Linear(hidden_dim+4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim,2))
        self.birth_pair_chunk=int(birth_pair_chunk)
        self.birth_pair_encoder=nn.Sequential(nn.Linear(3*hidden_dim+16+8,hidden_dim),nn.LayerNorm(hidden_dim),
                                              nn.GELU(),nn.Linear(hidden_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,1))

    def configuration(self):
        return {"schema": MODEL_SCHEMA, "visual_dim": self.visual_dim,
                "hidden_dim": self.hidden_dim, "geometry_dim": self.geometry_dim,
                "birth_pair_chunk":self.birth_pair_chunk,"birth_geometry_dim":16}

    def forward(self, history_features, observation_features, pair_geometry, observation_context,
                history_context,birth_pair_geometry=None,birth_pair_mask=None, *, use_geometry=True):
        m, n = len(history_features), len(observation_features)
        if history_features.shape != (m,self.visual_dim) or observation_features.shape != (n,self.visual_dim):
            raise ValueError("GMT_FEATURE_DIMENSION_MISMATCH")
        if pair_geometry.shape != (m,n,self.geometry_dim):
            raise ValueError("PAIR_GEOMETRY_SHAPE_MISMATCH")
        q, d = self.roi_projection(history_features), self.roi_projection(observation_features)
        qq, dd = q[:,None,:].expand(m,n,-1), d[None,:,:].expand(m,n,-1)
        geometry = pair_geometry if use_geometry else torch.zeros_like(pair_geometry)
        context_q = history_context[:,None,:].expand(m,n,-1)
        context_d = observation_context[None,:,:].expand(m,n,-1)
        pair = self.pair_encoder(torch.cat([qq, dd, torch.abs(qq-dd), geometry, context_q, context_d], dim=-1))
        life = self.lifecycle(torch.cat([q, history_context], dim=-1))
        delta = self.memory(pair)
        assignment_logits=self.assignment(pair).squeeze(-1)
        birth_history=d.new_zeros((n,4))
        if m and n:
            birth_history[:,0]=1.
            birth_history[:,1]=(history_features@observation_features.T).max(dim=0).values
            birth_history[:,2]=torch.tanh(assignment_logits.max(dim=0).values)
            if use_geometry:birth_history[:,3]=(geometry[:,:,6]*geometry[:,:,11]).max(dim=0).values
        birth_logits=self.birth(torch.cat([d,observation_context,birth_history],dim=-1)).squeeze(-1)
        if m==0:birth_logits=birth_logits+2.0  # no historical object can explain a current detection
        if birth_pair_geometry is None:birth_pair_geometry=d.new_zeros((n,n,16))
        if birth_pair_mask is None:
            camera_context=observation_context[:,2:]
            birth_pair_mask=torch.triu((camera_context[:,None]!=camera_context[None,:]).any(-1),diagonal=1)
        if birth_pair_geometry.shape!=(n,n,16) or birth_pair_mask.shape!=(n,n):
            raise ValueError("BIRTH_PAIR_SHAPE_MISMATCH")
        edges=birth_pair_mask.nonzero(as_tuple=False)
        parts=[]
        for start in range(0,len(edges),self.birth_pair_chunk):
            block=edges[start:start+self.birth_pair_chunk];left,right=block[:,0],block[:,1]
            a,b=d[left],d[right];ca,cb=observation_context[left],observation_context[right]
            bg=birth_pair_geometry[left,right] if use_geometry else d.new_zeros((len(block),16))
            interaction=torch.cat([a+b,torch.abs(a-b),a*b,bg,ca+cb,torch.abs(ca-cb)],dim=-1)
            parts.append(self.birth_pair_encoder(interaction).squeeze(-1))
        pair_logits=d.new_full((n,n),-30.)+d.sum()*0
        if parts:
            scores=torch.cat(parts)
            pair_logits=pair_logits.index_put((edges[:,0],edges[:,1]),scores)
            pair_logits=pair_logits.index_put((edges[:,1],edges[:,0]),scores)
        return {"assignment_logits": assignment_logits,
                "null_logits": life[:,0], "existence_logits": life[:,1],
                "birth_logits": birth_logits,
                "update_gate": torch.sigmoid(self.update(pair).squeeze(-1)),
                "memory_delta": delta,
                "birth_pair_logits":pair_logits,"birth_pair_mask":birth_pair_mask,
                "corrected_memory": F.normalize(observation_features[None,:,:] + 0.1*torch.tanh(delta), dim=-1)}
