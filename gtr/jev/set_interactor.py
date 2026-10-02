"""Permutation-equivariant option-set interaction blocks."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class SetBlock(nn.Module):
    def __init__(self, dim: int = 128, heads: int = 4, ffn_dim: int = 256) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim))

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        normalized = self.norm_attn(values)
        update, _ = self.attn(normalized, normalized, normalized,
                              key_padding_mask=~mask, need_weights=False)
        values = values + update
        values = values + self.ffn(self.norm_ffn(values))
        return values.masked_fill(~mask.unsqueeze(-1), 0.0)


class OptionSetInteractor(nn.Module):
    def __init__(self, dim: int = 128, layers: int = 2, heads: int = 4, ffn_dim: int = 256) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(SetBlock(dim, heads, ffn_dim) for _ in range(layers))

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        values = values.masked_fill(~mask.unsqueeze(-1), 0.0)
        for block in self.blocks:
            values = block(values, mask)
        return values
