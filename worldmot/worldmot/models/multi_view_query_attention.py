"""Joint query/observation attention for PWQT.

The module receives all camera observations as separate tokens.  It never
constructs a fixed cross-camera match before attention; a query may attend to
one observation, two observations, or the null token independently per view.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class QueryAttentionOutput:
    query_features: torch.Tensor
    observation_features: torch.Tensor
    attention: torch.Tensor
    geometry_bias: torch.Tensor


class MultiViewQueryAttention(nn.Module):
    def __init__(self, hidden_dim: int = 256, heads: int = 4, layers: int = 2,
                 dropout: float = 0.1, geometry_dim: int = 18):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.layers = int(layers)
        self.query_self = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
            for _ in range(layers)
        ])
        self.norm_self = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.q_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)])
        self.k_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)])
        self.v_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)])
        self.out_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)])
        self.ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, 4 * hidden_dim), nn.GELU(), nn.Linear(4 * hidden_dim, hidden_dim))
            for _ in range(layers)
        ])
        self.norm_cross = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.norm_ffn = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.geometry_bias = nn.Sequential(nn.Linear(geometry_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, heads))
        self.geometry_query = nn.Linear(6, hidden_dim)
        self.geometry_observation = nn.Linear(6, hidden_dim)

    def _cross(self, q: torch.Tensor, obs: torch.Tensor, obs_geometry: torch.Tensor,
               query_geometry: torch.Tensor, layer: int, obs_mask: torch.Tensor | None):
        b, nq, d = q.shape
        no = obs.shape[1]
        if no == 0:
            return torch.zeros_like(q), q.new_zeros((b, self.heads, nq, 0)), q.new_zeros((b, self.heads, nq, 0))
        qh = self.q_proj[layer](q).view(b, nq, self.heads, d // self.heads).transpose(1, 2)
        kh = self.k_proj[layer](obs).view(b, no, self.heads, d // self.heads).transpose(1, 2)
        vh = self.v_proj[layer](obs).view(b, no, self.heads, d // self.heads).transpose(1, 2)
        logits = torch.matmul(qh, kh.transpose(-2, -1)) / (d // self.heads) ** 0.5
        qb = query_geometry[:, :, :6]
        ob = obs_geometry[:, :, :6]
        # Geometry-conditioned bias is learned from relative position,
        # uncertainty and velocity channels.  It is a soft bias, not an ID
        # assignment or a nearest-neighbour merge.
        rel = ob[:, None, :, :] - qb[:, :, None, :]
        rel_full = torch.cat([rel, torch.zeros_like(rel[..., : max(0, obs_geometry.shape[-1] - 12)])], dim=-1)
        if rel_full.shape[-1] != self.geometry_bias[0].in_features:
            if rel_full.shape[-1] < self.geometry_bias[0].in_features:
                rel_full = torch.nn.functional.pad(rel_full, (0, self.geometry_bias[0].in_features - rel_full.shape[-1]))
            else:
                rel_full = rel_full[..., : self.geometry_bias[0].in_features]
        gb = self.geometry_bias(rel_full).permute(0, 3, 1, 2)
        logits = logits + gb
        if obs_mask is not None:
            logits = logits.masked_fill(~obs_mask[:, None, None, :], -1e4)
        weights = torch.softmax(logits, dim=-1)
        if obs_mask is not None:
            weights = weights * obs_mask[:, None, None, :]
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        attended = torch.matmul(weights, vh).transpose(1, 2).reshape(b, nq, d)
        return self.out_proj[layer](attended), weights, gb

    def forward(self, queries: torch.Tensor, observations: torch.Tensor,
                observation_geometry: torch.Tensor, query_geometry: torch.Tensor,
                observation_mask: torch.Tensor | None = None) -> QueryAttentionOutput:
        q = queries
        obs = observations
        all_weights = []
        all_bias = []
        for idx in range(self.layers):
            self_attended, _ = self.query_self[idx](q, q, q)
            q = self.norm_self[idx](q + self_attended)
            cross, weights, bias = self._cross(q, obs, observation_geometry, query_geometry, idx, observation_mask)
            q = self.norm_cross[idx](q + cross)
            q = self.norm_ffn[idx](q + self.ffn[idx](q))
            all_weights.append(weights)
            all_bias.append(bias)
        return QueryAttentionOutput(q, obs, torch.stack(all_weights, dim=1), torch.stack(all_bias, dim=1))
