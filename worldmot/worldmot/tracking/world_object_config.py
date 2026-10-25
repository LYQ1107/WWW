"""Distinct registered V3 methods; no implicit no-geometry interpretation."""
from __future__ import annotations
from dataclasses import dataclass, asdict

METHODS = {
    "FULL_WORLD_OBJECT_V3": (True, True, True),
    "HEAD_ONLY_NO_GEOMETRY_V3": (True, False, True),
    "FULL_NO_GEOMETRY_V3": (False, False, True),
    "APPEARANCE_CAPACITY_MATCHED_V3": (False, False, True),  # deprecated alias; no separate run
    "GEOMETRY_ONLY_ABLATION_V3": (True, True, False),
    "NO_LEARNED_JOINT_BIRTH_V3": (True, True, True),
    "RULE_BASED_WORLD_OBJECT_V3": (True,True,True),
}
METHOD_ALIASES={"APPEARANCE_CAPACITY_MATCHED_V3":"FULL_NO_GEOMETRY_V3"}


@dataclass(frozen=True)
class WorldObjectConfig:
    feature_dim: int
    method_id: str = "FULL_WORLD_OBJECT_V3"
    top_k: int = 4
    beam_width: int = 64
    max_exact_states: int = 4096
    max_observations: int = 1024
    max_history: int = 1024
    max_model_pairs: int = 262144
    max_birth_pair_edges: int = 262144
    birth_pair_chunk: int = 4096
    birth_pair_threshold: float = 0.7
    birth_pair_weight: float = 1.0
    max_missed: int = 30
    min_hits: int = 2
    min_memory_score: float = 0.5
    memory_momentum: float = 0.85
    min_appearance_similarity: float = 0.55
    recovery_similarity: float = 0.95
    joint_birth_similarity: float = 0.9
    geometry_gate: float = 6.0
    learned_weight: float = 0.75
    learned_identity_recovery_logit: float = 2.0
    birth_threshold: float = 0.5
    death_threshold: float = 0.1
    # None retains the exact R2 defaults. These are four separate interventions.
    enable_sync_geometry: bool | None = None
    enable_temporal_geometry: bool | None = None
    enable_spatial_state_update: bool | None = None
    enable_head_geometry: bool | None = None
    geometry_policy: str = "LEGACY_HARD"
    enable_motion_prediction: bool = False
    enable_safe_identity_memory: bool = False
    enable_multiview_coordination: bool = False
    enable_omega_scheduler: bool = False
    timestamp_ticks_per_second: float | None = None
    motion_acceleration_ratio: float = .01
    geometry_stale_ticks: float = 30.
    temporal_geometry_weight: float = .5
    identity_competition_margin: float = .05
    identity_learned_margin: float = 2.

    def __post_init__(self):
        if self.method_id in METHOD_ALIASES:
            object.__setattr__(self,"method_id",METHOD_ALIASES[self.method_id])
        if self.method_id not in METHODS or not isinstance(self.feature_dim,int) or self.feature_dim <= 0:
            raise ValueError("Unknown method or feature dimension missing from detector manifest")
        for name in ("top_k", "beam_width", "max_exact_states", "max_observations", "max_history", "max_model_pairs", "max_birth_pair_edges", "birth_pair_chunk", "max_missed", "min_hits"):
            if getattr(self, name) < 1:
                raise ValueError(f"invalid bound {name}")
        if not 0<=self.learned_weight<=1:
            raise ValueError("LEARNED_WEIGHT_MUST_BE_A_CONVEX_BLEND")
        if self.geometry_policy not in {"OFF", "LEGACY_HARD", "UNCERTAINTY_SOFT", "SPATIAL_ABSTAIN"}:
            raise ValueError("UNKNOWN_GEOMETRY_POLICY")
        if (self.motion_acceleration_ratio < 0 or self.geometry_stale_ticks <= 0
                or (self.timestamp_ticks_per_second is not None and self.timestamp_ticks_per_second <= 0)):
            raise ValueError("INVALID_MOTION_TIME_OR_NOISE")

    @property
    def sync_geometry(self):
        return METHODS[self.method_id][0] if self.enable_sync_geometry is None else self.enable_sync_geometry

    @property
    def temporal_geometry(self):
        return METHODS[self.method_id][0] if self.enable_temporal_geometry is None else self.enable_temporal_geometry

    @property
    def spatial_state_update(self):
        return METHODS[self.method_id][0] if self.enable_spatial_state_update is None else self.enable_spatial_state_update

    @property
    def use_geometry(self):
        return any((self.sync_geometry,self.temporal_geometry,self.spatial_state_update,self.head_geometry))

    @property
    def head_geometry(self):
        return METHODS[self.method_id][1] if self.enable_head_geometry is None else self.enable_head_geometry

    @property
    def use_appearance(self):
        return METHODS[self.method_id][2]

    @property
    def use_learned_joint_birth(self):
        return self.method_id!="NO_LEARNED_JOINT_BIRTH_V3"

    @property
    def checkpoint_method_id(self):
        return "FULL_WORLD_OBJECT_V3" if self.method_id=="NO_LEARNED_JOINT_BIRTH_V3" else self.method_id

    def to_dict(self):
        return asdict(self)
