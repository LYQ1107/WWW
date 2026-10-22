"""Persistent World Query Transformer (PWQT) core.

This is a trainable query state machine.  ``slot_id`` is a tensor index used
by the decoder; ``global_ids`` are runtime identities and are never inferred
from slot indices.  The model is intentionally usable with frozen RGB/Ω
features and exposes all assignment logits for supervision and audit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .multi_view_query_attention import MultiViewQueryAttention
from .world_query_decoder import WorldQueryDecoder


@dataclass
class QueryBatchState:
    embedding: torch.Tensor
    xyz: torch.Tensor
    velocity: torch.Tensor
    covariance: torch.Tensor
    appearance_memory: torch.Tensor
    existence: torch.Tensor
    visibility: torch.Tensor
    geometry: torch.Tensor
    age: torch.Tensor
    missed: torch.Tensor
    last_seen: torch.Tensor
    global_ids: torch.Tensor
    slot_ids: torch.Tensor

    def detach(self) -> "QueryBatchState":
        values = {}
        for key, value in self.__dict__.items():
            values[key] = value.detach() if torch.is_tensor(value) else value
        return QueryBatchState(**values)


@dataclass
class PersistentWorldQuery:
    slot_id: int
    global_id: int
    embedding: torch.Tensor
    xyz: torch.Tensor
    velocity: torch.Tensor
    covariance: torch.Tensor
    appearance_memory: torch.Tensor
    existence_score: torch.Tensor
    visibility: torch.Tensor
    age: int
    missed: int
    last_seen: int


class PersistentWorldQueryTransformer(nn.Module):
    def __init__(self, observation_dim: int = 340, hidden_dim: int = 256,
                 max_world_queries: int = 128, cameras: int = 2,
                 heads: int = 4, decoder_layers: int = 2, geometry_dim: int = 18):
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_world_queries = int(max_world_queries)
        self.cameras = int(cameras)
        self.geometry_dim = int(geometry_dim)
        self.slot_embeddings = nn.Parameter(torch.randn(max_world_queries, hidden_dim) * 0.02)
        # Camera IDs are an input condition, not an identity label.  A few
        # spare entries keep IDs 1/2 valid while allowing camera-order
        # permutation tests without hard-coding a training scene number.
        self.camera_embeddings = nn.Embedding(max(8, self.cameras + 1), hidden_dim)
        self.observation_projection = nn.Sequential(nn.LayerNorm(observation_dim), nn.Linear(observation_dim, hidden_dim), nn.GELU())
        self.query_geometry_projection = nn.Linear(geometry_dim, hidden_dim)
        self.attention = MultiViewQueryAttention(hidden_dim, heads, decoder_layers, geometry_dim=geometry_dim)
        self.decoder = WorldQueryDecoder(hidden_dim, cameras, geometry_dim)

    def init_state(self, batch_size: int, device=None, dtype=None) -> QueryBatchState:
        device = device or self.slot_embeddings.device
        dtype = dtype or self.slot_embeddings.dtype
        q = self.slot_embeddings.to(device=device, dtype=dtype)[None].expand(batch_size, -1, -1).clone()
        z3 = torch.zeros(batch_size, self.max_world_queries, 3, device=device, dtype=dtype)
        zc = torch.eye(3, device=device, dtype=dtype)[None, None].expand(batch_size, self.max_world_queries, -1, -1).clone()
        zg = torch.zeros(batch_size, self.max_world_queries, self.geometry_dim, device=device, dtype=dtype)
        return QueryBatchState(
            embedding=q, xyz=z3.clone(), velocity=z3.clone(), covariance=zc,
            appearance_memory=q.clone(), existence=torch.zeros(batch_size, self.max_world_queries, device=device, dtype=dtype),
            visibility=torch.zeros(batch_size, self.max_world_queries, self.cameras, device=device, dtype=dtype),
            geometry=zg, age=torch.zeros(batch_size, self.max_world_queries, device=device, dtype=torch.long),
            missed=torch.zeros(batch_size, self.max_world_queries, device=device, dtype=torch.long),
            last_seen=torch.full((batch_size, self.max_world_queries), -1, device=device, dtype=torch.long),
            global_ids=torch.full((batch_size, self.max_world_queries), -1, device=device, dtype=torch.long),
            slot_ids=torch.arange(self.max_world_queries, device=device, dtype=torch.long)[None].expand(batch_size, -1).clone(),
        )

    def forward(self, observations: torch.Tensor, observation_geometry: torch.Tensor,
                state: QueryBatchState | None = None, observation_mask: torch.Tensor | None = None,
                camera_ids: torch.Tensor | None = None) -> dict:
        if observations.ndim != 3 or observation_geometry.ndim != 3:
            raise ValueError("observations and observation_geometry must be BxNxD")
        b = observations.shape[0]
        if state is None:
            state = self.init_state(b, observations.device, observations.dtype)
        obs = self.observation_projection(observations)
        if camera_ids is not None:
            camera_ids = camera_ids.to(device=observations.device, dtype=torch.long)
            if camera_ids.ndim == 1:
                camera_ids = camera_ids[None]
            camera_ids = camera_ids.clamp_min(0).clamp_max(self.camera_embeddings.num_embeddings - 1)
            obs = obs + self.camera_embeddings(camera_ids)
        # Carry the previous query geometry into the next decoder update.  The
        # projection is learned and remains in the query state; it is not a
        # hand-written distance matcher.
        query_input = state.embedding + self.query_geometry_projection(state.geometry)
        attn = self.attention(query_input, obs, observation_geometry, state.geometry, observation_mask)
        decoded = self.decoder(attn.query_features, obs)
        new_geometry = decoded["geometry"]
        new_state = QueryBatchState(
            embedding=attn.query_features,
            xyz=decoded["xyz"], velocity=decoded["velocity"],
            covariance=torch.diag_embed(torch.nn.functional.softplus(decoded["log_covariance"]) + 1e-4),
            appearance_memory=attn.query_features,
            existence=torch.sigmoid(decoded["existence_logits"]),
            visibility=torch.sigmoid(decoded["visibility_logits"]), geometry=new_geometry,
            age=state.age + 1, missed=state.missed, last_seen=state.last_seen,
            global_ids=state.global_ids, slot_ids=state.slot_ids,
        )
        return {**decoded, "queries": attn.query_features, "attention": attn.attention,
                "geometry_bias": attn.geometry_bias, "state": new_state,
                "previous_state": state, "observation_features": obs}
