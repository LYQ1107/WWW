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
                 heads: int = 4, decoder_layers: int = 2, geometry_dim: int = 18,
                 ablation: str = "joint"):
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_world_queries = int(max_world_queries)
        self.cameras = int(cameras)
        self.geometry_dim = int(geometry_dim)
        self.ablation = ablation
        self.config = dict(observation_dim=observation_dim, hidden_dim=hidden_dim,
                           max_world_queries=max_world_queries, cameras=cameras,
                           heads=heads, decoder_layers=decoder_layers, geometry_dim=geometry_dim,
                           ablation=ablation)
        self.slot_embeddings = nn.Parameter(torch.randn(max_world_queries, hidden_dim) * 0.02)
        # Camera IDs are an input condition, not an identity label.  A few
        # spare entries keep IDs 1/2 valid while allowing camera-order
        # permutation tests without hard-coding a training scene number.
        self.camera_embeddings = nn.Embedding(max(8, self.cameras + 1), hidden_dim)
        self.observation_projection = nn.Sequential(nn.LayerNorm(observation_dim), nn.Linear(observation_dim, hidden_dim), nn.GELU())
        self.query_geometry_projection = nn.Linear(geometry_dim, hidden_dim)
        self.attention = MultiViewQueryAttention(hidden_dim, heads, decoder_layers, geometry_dim=geometry_dim)
        self.decoder = WorldQueryDecoder(hidden_dim, cameras, geometry_dim)
        self.rgb_projection = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, hidden_dim), nn.GELU())
        self.detector_projection = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, hidden_dim), nn.GELU())
        self.omega_projection = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, hidden_dim), nn.GELU())
        self.appearance_fusion = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.LayerNorm(hidden_dim))
        self.observed_geometry_projection = nn.Linear(geometry_dim + 2, hidden_dim)
        self.update_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.identity_projection = nn.Linear(hidden_dim, hidden_dim)
        self.pair_head = nn.Sequential(nn.Linear(2 * hidden_dim + 6, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.match_geometry_head = nn.Sequential(nn.Linear(6, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
        self.logit_scale = nn.Parameter(torch.tensor(2.0))
        self.birth_novelty = nn.Linear(1, 1)
        # A fixed novelty prior distinguishes an empty scene memory from a
        # previously observed identity; training learns the final calibration.
        nn.init.constant_(self.birth_novelty.weight, 2.0)
        nn.init.constant_(self.birth_novelty.bias, -1.0)

    def encode_observations(self, observations, geometry):
        if self.observation_dim != 532:
            return self.observation_projection(observations)
        rgb = self.rgb_projection(observations[..., :128])
        det = self.detector_projection(observations[..., 128:256])
        omega = self.omega_projection(observations[..., 256:512])
        if self.ablation == "no_omega":
            omega = omega * 0
        if self.ablation == "no_appearance":
            rgb, det, omega = rgb * 0, det * 0, omega * 0
        visual = self.appearance_fusion(torch.cat([rgb, det, omega], -1))
        g = geometry if self.ablation != "no_omega" else geometry * 0
        auxiliary = observations[..., -2:]
        if self.ablation == "no_omega":
            auxiliary = torch.cat([auxiliary[..., :1], auxiliary[..., 1:] * 0], -1)
        return visual + self.observed_geometry_projection(torch.cat([g, auxiliary], -1))

    def accept_observations(self, state, output, assignments, world_xyz=None, world_covariance=None, geometry_valid=None):
        """Initialize births and update recurrent memory from actual observations.

        assignments is one list of {slot: observation_indices} per batch.
        This differentiable state update is shared by teacher-forced BPTT
        and causal inference; identity labels are never embedded in a query.
        """
        obs = output["observation_features"]
        weights = obs.new_zeros((obs.shape[0], self.max_world_queries, obs.shape[1]))
        for b, mapping in enumerate(assignments):
            for q, indices in mapping.items():
                if indices: weights[b, q, indices] = 1.0 / len(indices)
        observed = weights.sum(-1) > 0
        pooled = weights @ obs
        newly_born = observed & (state.global_ids < 0)
        recurrent = self.update_cell(pooled.reshape(-1, self.hidden_dim), state.embedding.reshape(-1, self.hidden_dim)).reshape_as(state.embedding)
        embedding = torch.where(newly_born[..., None], pooled, recurrent)
        state.embedding = torch.where(observed[..., None], embedding, state.embedding)
        appearance = torch.where(newly_born[..., None], pooled,
                                 0.8 * state.appearance_memory + 0.2 * pooled)
        state.appearance_memory = torch.where(observed[..., None], appearance, state.appearance_memory)
        state.age = torch.where(newly_born, torch.ones_like(state.age), state.age)
        gweights = weights
        if geometry_valid is not None:
            gweights = weights * geometry_valid[:, None, :].float()
            gweights = gweights / gweights.sum(-1, keepdim=True).clamp_min(1e-8)
        geometry_observed = gweights.sum(-1) > 0
        state.geometry = torch.where(geometry_observed[..., None], gweights @ output["input_geometry"], state.geometry)
        if world_xyz is not None:
            xyz = gweights @ torch.nan_to_num(world_xyz)
            delta = torch.where(newly_born[..., None], torch.zeros_like(xyz), xyz - state.xyz)
            state.velocity = torch.where(geometry_observed[..., None], delta, state.velocity)
            state.xyz = torch.where(geometry_observed[..., None], xyz, state.xyz)
        if world_covariance is not None:
            cov = (gweights @ torch.nan_to_num(world_covariance).flatten(-2)).reshape_as(state.covariance)
            state.covariance = torch.where(geometry_observed[..., None, None], cov, state.covariance)
        state.missed = torch.where(observed, torch.zeros_like(state.missed), state.missed + 1)
        state.existence = torch.where(observed, torch.ones_like(state.existence), state.existence)
        return state

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
        if self.ablation == "no_omega": observation_geometry = observation_geometry * 0
        obs = self.encode_observations(observations, observation_geometry)
        visual_obs = obs
        no = obs.shape[1]
        left = visual_obs[:, :, None, :].expand(-1, -1, no, -1)
        right = visual_obs[:, None, :, :].expand(-1, no, -1, -1)
        pair_geometry = observation_geometry[:, :, None, :6] - observation_geometry[:, None, :, :6]
        pair_logits = self.pair_head(torch.cat([abs(left-right), left*right, abs(pair_geometry)], -1)).squeeze(-1)
        if self.ablation == "two_stage" and no:
            # Explicit learned stage 1: soft cross-camera fusion precedes the
            # temporal query decoder.  All methods retain the same detections,
            # feature sources and loss/epoch budget.  Joint uses no pre-fusion.
            pair_weights = pair_logits.sigmoid()
            if camera_ids is not None:
                pair_weights = pair_weights * (camera_ids[:, :, None] != camera_ids[:, None, :])
            if observation_mask is not None:
                pair_weights = pair_weights * observation_mask[:, None, :]
            eye = torch.eye(no, device=obs.device, dtype=obs.dtype)[None]
            pair_weights = pair_weights + eye
            pair_weights = pair_weights / pair_weights.sum(-1, keepdim=True).clamp_min(1e-8)
            obs = pair_weights @ visual_obs
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
        # A recurrent appearance memory gives the learned scorer an identity
        # anchor rather than relying on a random slot embedding.
        qkey = torch.nn.functional.normalize(self.identity_projection(state.appearance_memory), dim=-1)
        okey = torch.nn.functional.normalize(self.identity_projection(visual_obs), dim=-1)
        similarity = qkey @ okey.transpose(-1, -2)
        rel = observation_geometry[:, None, :, :6] - state.geometry[:, :, None, :6]
        geo_match = self.match_geometry_head(rel).squeeze(-1)
        if self.ablation == "no_omega": geo_match = geo_match * 0
        decoded["assignment_logits"] = torch.cat([
            decoded["assignment_logits"][..., :-1] + self.logit_scale.exp().clamp_max(20) * similarity + geo_match,
            decoded["assignment_logits"][..., -1:]], -1)
        active = state.global_ids >= 0
        novelty = similarity.masked_fill(~active[:, :, None], -1.0).amax(1) if similarity.shape[-1] else similarity.new_empty((b, 0))
        decoded["birth_logits"] = decoded["birth_logits"] + self.birth_novelty((1 - novelty)[..., None]).squeeze(-1)
        decoded["pair_logits"] = pair_logits
        new_state = QueryBatchState(
            embedding=attn.query_features,
            xyz=state.xyz, velocity=state.velocity, covariance=state.covariance,
            appearance_memory=state.appearance_memory,
            existence=torch.sigmoid(decoded["existence_logits"]),
            visibility=torch.sigmoid(decoded["visibility_logits"]), geometry=state.geometry,
            age=state.age + 1, missed=state.missed, last_seen=state.last_seen,
            global_ids=state.global_ids, slot_ids=state.slot_ids,
        )
        return {**decoded, "queries": attn.query_features, "attention": attn.attention,
                "geometry_bias": attn.geometry_bias, "state": new_state,
                "previous_state": state, "observation_features": visual_obs,
                "input_geometry": observation_geometry}
