"""World-object measurements are formed only AFTER joint identity assignment."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .world_object_observation import fuse_identity_features
from .world_object_support import WorldObjectSupport


@dataclass
class WorldObjectHypothesis:
    object_id: int
    observations: list
    support: WorldObjectSupport | None
    score: float
    status: str
    feature: np.ndarray | None
    feature_valid: bool
    diagnostics: dict = field(default_factory=dict)

    @property
    def cameras(self):
        return tuple(sorted(o.camera_id for o in self.observations))

    @property
    def center(self):
        return self.support.center if self.support is not None else np.full(3, np.nan)


def assigned_measurement(observations, object_id, feature_dim, use_geometry=True, coordinate_views=False):
    observations = list(observations)
    if len({o.camera_id for o in observations}) != len(observations):
        raise ValueError("ONE_OBSERVATION_PER_CAMERA_PER_OBJECT_REQUIRED")
    if coordinate_views:
        usable=sorted((o for o in observations if o.visible and o.feature_valid),key=lambda o:(-o.score,o.key))
        feature=usable[0].roi_feature.copy() if usable else None
    else:feature, _ = fuse_identity_features(observations, feature_dim, use_geometry=use_geometry)
    geometry = [o for o in observations if use_geometry and o.visible and o.world_state_update_eligible]
    support = None
    if geometry:
        weights = np.array([o.score * o.geometry_reliability for o in geometry])
        center = np.average([o.world_center for o in geometry], axis=0, weights=weights)
        cov = sum(w * (o.world_covariance + np.outer(o.world_center - center, o.world_center - center))
                  for w, o in zip(weights, geometry)) / weights.sum()
        points = [o.spatial_support.foreground_points for o in geometry
                  if o.spatial_support is not None and len(o.spatial_support.foreground_points)]
        foreground = np.concatenate(points) if points else np.empty((0, 3))
        support = WorldObjectSupport(foreground, foreground, center, cov,
                                     float(np.average([o.geometry_reliability for o in geometry], weights=weights)),
                                     1.0, "POST_ASSIGNMENT_MEASUREMENT", coordinate_frame="PERSISTENT_WORLD",
                                     depth_source="VERIFIED_CAMERA_DEPTH")
    return WorldObjectHypothesis(object_id, observations, support,
                                 max((o.score for o in observations), default=0.0),
                                 "ASSIGNED_MULTI_VIEW" if len(observations) > 1 else "ASSIGNED_SINGLE_VIEW",
                                 feature, feature is not None)


class JointWorldObjectBuilder:
    """Diagnostic facade; formal joint assignment requires historical objects.

    The old greedy implementation is reproducible at source commit 190c735.
    This facade emits singletons and cannot commit camera pairs in advance.
    """
    def __init__(self, **kwargs):
        self.diagnostic_settings = kwargs

    def build(self, observations, start_id=1):
        obs = sorted(observations, key=lambda o: o.key)
        dim = next((o.roi_feature.size for o in obs if o.feature_valid), None)
        hypotheses = [assigned_measurement([o], start_id + i, dim) for i, o in enumerate(obs)]
        return hypotheses, {"accepted_pairs": 0, "single_view_objects": len(obs),
                            "premerge_disabled": True, "requires_history_joint_assignment": True}
