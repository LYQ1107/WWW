"""Fixed V5 variants. Membership never controls birth, existence or memory writes."""
from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass(frozen=True)
class WorldObjectV5Config:
    variant: str = "B3"
    feature_dim: int = 1024
    hidden_dim: int = 256
    geometry_dim: int = 32
    top_k: int = 4
    beam_width: int = 64
    max_exact_states: int = 4096
    solver_expansion_budget: int = 200000
    membership_min_margin: float = 0.
    low_score_recovery_min_margin: float = .5
    identity_write_min_margin: float = 1.
    identity_write_competition_margin: float = .5
    memory_momentum: float = .85
    confirmed_lost_retention_ticks: int = 90
    tentative_lost_retention_ticks: int = 2
    min_confirmation_hits: int = 2
    geometry_stale_ticks: float = 30.
    geometry_bonus_bound: float = .5
    alignment_min_inlier_ratio: float = .25
    reliable_geometry_min_quality: float = .6
    geometry_pair_max_mahalanobis: float = 6.
    birth_geometry_min_roi_cosine: float = .65
    birth_visual_only_min_roi_cosine: float = .97
    birth_pair_competition_margin: float = .03
    birth_min_reprojection_coverage: float = .3
    duplicate_min_iou: float = .95
    duplicate_min_roi_cosine: float = .995
    duplicate_min_containment: float = .98
    birth_high_score: float = .7
    pair_chunk: int = 4096
    feature_dropout: float = .05

    def __post_init__(self):
        if self.variant not in {"B1", "B2", "B3", "B4"}:
            raise ValueError("B0 uses the unchanged frozen V4 C1 core")
        if self.feature_dim != 1024 or self.geometry_dim != 32:
            raise ValueError("Registered GMT descriptor and equal-capacity geometry contract required")

    @property
    def use_geometry(self):
        return self.variant != "B4"

    @property
    def rich_geometry(self):
        return self.variant == "B3"

    @property
    def frozen_legacy_membership(self):
        return self.variant == "B1"

    def to_dict(self):
        return asdict(self)


def registered_config(variant, calibration=None):
    path=Path(__file__).resolve().parents[2]/"configs/worldobject_v5/registered.json"
    registered=json.loads(path.read_text())
    keys=set(WorldObjectV5Config.__dataclass_fields__)
    values={k:v for k,v in registered.items() if k in keys}
    values["variant"]=variant
    if calibration is not None:
        if calibration.get("split")!="train" or calibration.get("calibration_scenes")!=registered["fit_scenes"]:
            raise ValueError("Birth calibration must use registered fit18 TRAIN only")
        values["birth_high_score"]=float(calibration["birth_high_score"])
    return WorldObjectV5Config(**values)
