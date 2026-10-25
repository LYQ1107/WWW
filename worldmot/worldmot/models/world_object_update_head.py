"""V3 history-conditioned identity assignment, lifecycle and memory correction."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

MODEL_SCHEMA = "worldobject_identity_assignment_v3"


class WorldObjectUpdateHead(nn.Module):
    """D comes from frozen GMT manifest, not from a synthetic bbox descriptor.

    Inputs: history[M,D], observations[N,D], geometry[M,N,12],
    observation_context[N,4], history_context[M,4]. No GT identity is an input.
    """
    def __init__(self, visual_dim, hidden_dim=128, geometry_dim=12):
        super().__init__()
        self.visual_dim, self.hidden_dim, self.geometry_dim = int(visual_dim), int(hidden_dim), int(geometry_dim)
        self.roi_projection = nn.Linear(self.visual_dim, hidden_dim)
        self.pair_encoder = nn.Sequential(nn.Linear(3*hidden_dim + geometry_dim + 8, hidden_dim),
                                          nn.LayerNorm(hidden_dim), nn.GELU(),
                                          nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.assignment = nn.Linear(hidden_dim, 1)
        self.update = nn.Linear(hidden_dim, 1)
        self.memory = nn.Linear(hidden_dim, self.visual_dim)
        self.birth = nn.Sequential(nn.Linear(hidden_dim+4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim,1))
        self.lifecycle = nn.Sequential(nn.Linear(hidden_dim+4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim,2))

    def configuration(self):
        return {"schema": MODEL_SCHEMA, "visual_dim": self.visual_dim,
                "hidden_dim": self.hidden_dim, "geometry_dim": self.geometry_dim}

    def forward(self, history_features, observation_features, pair_geometry, observation_context,
                history_context, *, use_geometry=True):
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
        return {"assignment_logits": self.assignment(pair).squeeze(-1),
                "null_logits": life[:,0], "existence_logits": life[:,1],
                "birth_logits": self.birth(torch.cat([d, observation_context], dim=-1)).squeeze(-1),
                "update_gate": torch.sigmoid(self.update(pair).squeeze(-1)),
                "memory_delta": delta,
                "corrected_memory": F.normalize(observation_features[None,:,:] + 0.1*torch.tanh(delta), dim=-1)}
