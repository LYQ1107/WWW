"""Prediction heads for the persistent world-query state."""

from __future__ import annotations

import torch
from torch import nn

from .query_birth_head import QueryBirthHead
from .query_projection_head import QueryProjectionHead
from .query_visibility_head import QueryVisibilityHead


class WorldQueryDecoder(nn.Module):
    def __init__(self, hidden_dim: int = 256, cameras: int = 2, geometry_dim: int = 18):
        super().__init__()
        self.assignment = nn.Linear(hidden_dim, hidden_dim)
        self.observation_key = nn.Linear(hidden_dim, hidden_dim)
        self.null_assignment = nn.Linear(hidden_dim, 1)
        self.xyz = nn.Linear(hidden_dim, 3)
        self.velocity = nn.Linear(hidden_dim, 3)
        self.covariance = nn.Linear(hidden_dim, 3)
        self.existence = nn.Linear(hidden_dim, 1)
        self.geometry = nn.Linear(hidden_dim, geometry_dim)
        self.visibility = QueryVisibilityHead(hidden_dim, cameras)
        self.birth = QueryBirthHead(hidden_dim)
        self.projection = QueryProjectionHead(hidden_dim)

    def forward(self, queries: torch.Tensor, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        q = self.assignment(queries)
        k = self.observation_key(observations)
        logits = torch.matmul(q, k.transpose(-1, -2)) / max(q.shape[-1], 1) ** 0.5
        null = self.null_assignment(queries)
        assignment_logits = torch.cat([logits, null], dim=-1)
        return {
            "assignment_logits": assignment_logits,
            "existence_logits": self.existence(queries).squeeze(-1),
            "visibility_logits": self.visibility(queries),
            "birth_logits": self.birth(observations),
            "xyz": self.xyz(queries),
            "velocity": self.velocity(queries),
            "log_covariance": self.covariance(queries),
            "geometry": self.geometry(queries),
            "projection_residual": self.projection(queries),
        }
