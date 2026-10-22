"""Camera-conditioned visibility head."""

from __future__ import annotations

import torch
from torch import nn


class QueryVisibilityHead(nn.Module):
    def __init__(self, hidden_dim: int = 256, cameras: int = 2):
        super().__init__()
        self.cameras = int(cameras)
        self.net = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, cameras))

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        return self.net(queries)
