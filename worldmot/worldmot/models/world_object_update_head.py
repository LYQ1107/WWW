"""Small learned update head for W4 (backbone and detector remain frozen)."""

from __future__ import annotations

import torch
from torch import nn


class WorldObjectUpdateHead(nn.Module):
    """Predict an update gate and compact memory delta for one world object.

    This is deliberately not a renamed V2 query-attention block: geometry is a
    persistent state input and the gate controls whether a measurement may
    rewrite that state.
    """

    def __init__(self, geometry_dim: int = 12, visual_dim: int = 256, hidden_dim: int = 128,
                 memory_dim: int = 64):
        super().__init__()
        self.geometry_dim = int(geometry_dim); self.visual_dim = int(visual_dim)
        self.memory_dim = int(memory_dim)
        self.encoder = nn.Sequential(nn.Linear(geometry_dim + visual_dim + memory_dim, hidden_dim),
                                     nn.LayerNorm(hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.update_gate = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.birth_logit = nn.Linear(hidden_dim, 1)
        self.memory_delta = nn.Linear(hidden_dim, memory_dim)

    def forward(self, geometry_state, visual_feature, memory_state):
        x = torch.cat([geometry_state, visual_feature, memory_state], dim=-1)
        h = self.encoder(x)
        return {"update_gate": self.update_gate(h).squeeze(-1),
                "birth_logit": self.birth_logit(h).squeeze(-1),
                "memory_delta": self.memory_delta(h)}
