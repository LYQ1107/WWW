"""Compact permutation-equivariant dynamic-candidate choice head."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .set_interactor import OptionSetInteractor


class SetChoiceHead(nn.Module):
    """Encode a query and a masked dynamic option set into full probabilities."""

    def __init__(self, query_dim: int, option_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.query_encoder = nn.Sequential(nn.Linear(query_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.option_encoder = nn.Sequential(nn.Linear(option_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.interactor = OptionSetInteractor(hidden, layers=2, heads=4, ffn_dim=256)
        self.query_score = nn.Linear(hidden, hidden, bias=False)
        self.option_score = nn.Linear(hidden, hidden, bias=False)
        self.hidden = hidden

    def forward(self, query: Tensor, options: Tensor, option_mask: Tensor) -> Tensor:
        if query.ndim != 2 or options.ndim != 3 or option_mask.shape != options.shape[:2]:
            raise ValueError("invalid query/options/mask shape")
        q = self.query_encoder(query.float())
        option = self.option_encoder(options.float())
        option = self.interactor(option + q.unsqueeze(1), option_mask)
        q = self.query_score(q).float()
        option = self.option_score(option).float()
        logits = (q.unsqueeze(1) * option).sum(-1) / math.sqrt(self.hidden)
        return logits.masked_fill(~option_mask, -torch.inf)


class IndependentLinear(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(input_dim, 1)

    def forward(self, query: Tensor, options: Tensor, option_mask: Tensor) -> Tensor:
        q = query.unsqueeze(1).expand(-1, options.shape[1], -1)
        logits = self.score(torch.cat([q, options], dim=-1)).squeeze(-1)
        return logits.masked_fill(~option_mask, -torch.inf)


class IndependentMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, query: Tensor, options: Tensor, option_mask: Tensor) -> Tensor:
        q = query.unsqueeze(1).expand(-1, options.shape[1], -1)
        logits = self.score(torch.cat([q, options], dim=-1)).squeeze(-1)
        return logits.masked_fill(~option_mask, -torch.inf)
