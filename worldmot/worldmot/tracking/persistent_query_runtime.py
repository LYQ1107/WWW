"""Online runtime for trained Persistent World Query models."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..models.persistent_world_queries import PersistentWorldQueryTransformer, QueryBatchState
from ..models.world_observation_encoder import WorldObservationToken, observations_to_tensors
from .query_lifecycle import constrained_assign
from .query_memory_bank import QueryMemoryBank


@dataclass
class RuntimeResult:
    timestamp: int
    status: str
    assignments: dict[int, int]
    global_ids_by_observation: dict[int, int]
    births: list[int]
    diagnostics: dict
    query_to_observations: dict[int, list[int]]


class PersistentQueryRuntime:
    def __init__(self, model: PersistentWorldQueryTransformer, device: str = "cpu",
                 checkpoint: str | None = None, allow_untrained: bool = False,
                 max_missed: int = 12):
        self.model = model.to(device).eval()
        self.device = str(device)
        self.checkpoint = checkpoint
        self.trained = False
        if checkpoint:
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            self.model.load_state_dict(state.get("model", state), strict=True)
            self.trained = True
        if not self.trained and not allow_untrained:
            raise RuntimeError("PWQT checkpoint is required for formal inference; random weights are smoke-only")
        self.allow_untrained = bool(allow_untrained)
        self.max_missed = int(max_missed)
        # The decoder currently predicts clip-normalized coordinates.  Until
        # a trained world-coordinate calibration head is supplied, projections
        # are diagnostic only and cannot enter MOT evaluation.
        self.coordinates_calibrated = False
        self.memory = QueryMemoryBank()
        self.state: QueryBatchState | None = None
        self.scene_id = None
        self.last_timestamp = None

    def reset(self, scene_id=None):
        self.scene_id = scene_id
        self.memory.reset()
        self.state = None
        self.last_timestamp = None

    @torch.inference_mode()
    def step(self, tokens: list[WorldObservationToken], timestamp: int) -> RuntimeResult:
        obs, geo, cameras, valid = observations_to_tensors(tokens, device=self.device)
        camera_tensor = torch.as_tensor(cameras, device=obs.device, dtype=torch.long)[None]
        if self.state is None:
            self.state = self.model.init_state(1, obs.device, obs.dtype)
        output = self.model(obs, geo, self.state, valid, camera_ids=camera_tensor)
        assignment_logits = output["assignment_logits"].clone()
        birth_logits = output["birth_logits"].clone()
        if valid.numel():
            invalid = ~valid[:, None, :]
            assignment_logits[..., :-1] = assignment_logits[..., :-1].masked_fill(invalid, -1e9)
            birth_logits = birth_logits.masked_fill(~valid, -1e9)
        assignment = constrained_assign(assignment_logits, birth_logits, cameras,
                                        output["state"].existence, birth_threshold=0.5)
        global_by_obs: dict[int, int] = {}
        births = []
        query_to_observations = {int(q): list(obs_indices)
                                 for q, obs_indices in assignment.query_to_observations.items()}
        for q, observations in query_to_observations.items():
            first_observation = observations[0]
            gid = int(output["state"].global_ids[0, q].item())
            if gid < 0:
                gid = self.memory.allocate(timestamp, first_observation)
                output["state"].global_ids[0, q] = gid
                births.append(gid)
            for o in observations:
                global_by_obs[o] = gid
            output["state"].last_seen[0, q] = int(timestamp)
            output["state"].missed[0, q] = 0
        for q in range(self.model.max_world_queries):
            if q not in query_to_observations and int(output["state"].global_ids[0, q]) >= 0:
                output["state"].missed[0, q] += 1
                if int(output["state"].missed[0, q]) > self.max_missed:
                    self.memory.close(int(output["state"].global_ids[0, q]), timestamp)
                    output["state"].global_ids[0, q] = -1
                    output["state"].existence[0, q] = 0.0
        for o in assignment.births:
            # A birth consumes an unused query slot.  It does not reuse a
            # numeric slot's old identity.
            free = [q for q in range(self.model.max_world_queries)
                    if int(output["state"].global_ids[0, q]) < 0 and q not in query_to_observations]
            if not free:
                continue
            q = free[0]
            gid = self.memory.allocate(timestamp, o)
            output["state"].global_ids[0, q] = gid
            output["state"].last_seen[0, q] = int(timestamp)
            output["state"].missed[0, q] = 0
            output["state"].existence[0, q] = 1.0
            global_by_obs[o] = gid
            births.append(gid)
            query_to_observations[q] = [o]
        self.state = output["state"].detach()
        self.last_timestamp = int(timestamp)
        status = "TRAINED" if self.trained else "SMOKE_UNTRAINED"
        return RuntimeResult(int(timestamp), status,
                             {q: obs[0] for q, obs in query_to_observations.items() if obs},
                             global_by_obs, births, {
            "scene_id": self.scene_id, "slot_ids_are_not_global_ids": True,
            "query_coordinates": "clip_normalized_decoder_output",
            "coordinates_calibrated": bool(self.coordinates_calibrated),
            "observation_count": len(tokens), "valid_observation_count": int(valid.sum().item()),
            "duplicate_rejections": assignment.duplicate_rejections,
            "unmatched_observations": assignment.unmatched_observations,
            "query_capacity": self.model.max_world_queries,
            "query_capacity_overflow": bool(len(assignment.births) > sum(int(x < 0) for x in self.state.global_ids[0].tolist())),
            "checkpoint": self.checkpoint, "random_weights_forbidden_formal": not self.trained,
        }, query_to_observations)
