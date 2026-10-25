"""Distinct registered V3 methods; no implicit no-geometry interpretation."""
from dataclasses import dataclass, asdict

METHODS = {
    "FULL_WORLD_OBJECT_V3": (True, True, True),
    "HEAD_ONLY_NO_GEOMETRY_V3": (True, False, True),
    "FULL_NO_GEOMETRY_V3": (False, False, True),
    "APPEARANCE_CAPACITY_MATCHED_V3": (False, False, True),
    "GEOMETRY_ONLY_ABLATION_V3": (True, True, False),
}


@dataclass(frozen=True)
class WorldObjectConfig:
    feature_dim: int
    method_id: str = "FULL_WORLD_OBJECT_V3"
    top_k: int = 4
    beam_width: int = 64
    max_exact_states: int = 4096
    max_observations: int = 1024
    max_history: int = 1024
    max_model_pairs: int = 65536
    max_missed: int = 30
    min_hits: int = 2
    min_memory_score: float = 0.5
    memory_momentum: float = 0.85
    min_appearance_similarity: float = 0.55
    recovery_similarity: float = 0.95
    joint_birth_similarity: float = 0.9
    geometry_gate: float = 6.0
    learned_weight: float = 0.75
    birth_threshold: float = 0.5
    death_threshold: float = 0.1

    def __post_init__(self):
        if self.method_id not in METHODS or not isinstance(self.feature_dim,int) or self.feature_dim <= 0:
            raise ValueError("Unknown method or feature dimension missing from detector manifest")
        for name in ("top_k", "beam_width", "max_exact_states", "max_observations", "max_history", "max_model_pairs", "max_missed", "min_hits"):
            if getattr(self, name) < 1:
                raise ValueError(f"invalid bound {name}")
        if not 0<=self.learned_weight<=1:
            raise ValueError("LEARNED_WEIGHT_MUST_BE_A_CONVEX_BLEND")

    @property
    def use_geometry(self):
        return METHODS[self.method_id][0]

    @property
    def head_geometry(self):
        return METHODS[self.method_id][1]

    @property
    def use_appearance(self):
        return METHODS[self.method_id][2]

    def to_dict(self):
        return asdict(self)
