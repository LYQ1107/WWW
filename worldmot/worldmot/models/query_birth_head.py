"""Newborn-object classification head."""

from __future__ import annotations

import torch
from torch import nn


class QueryBirthHead(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations).squeeze(-1)
