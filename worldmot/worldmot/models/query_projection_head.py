"""Projection head contract for query-to-camera output."""

from __future__ import annotations

import torch
from torch import nn


class QueryProjectionHead(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.box = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 4))

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        # The output is a learned residual/extent proposal.  It is not an
        # evaluation-eligible observation until a current target-view mask or
        # detector assignment is attached by the runtime.
        return self.box(queries)
