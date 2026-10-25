"""GT-free observation contract shared by live, replay and TRAIN forward passes."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

OBSERVATION_SCHEMA = "worldobject_observation_v3_birthfix_v1"
GEOMETRY_VALID = "GEOMETRY_VALID"
GEOMETRY_LOW_CONFIDENCE = "GEOMETRY_LOW_CONFIDENCE"
WORLD_ALIGNMENT_FAILED = "WORLD_ALIGNMENT_FAILED"
SPATIAL_SUPPORT_AMBIGUOUS = "SPATIAL_SUPPORT_AMBIGUOUS"
VISUAL_ONLY_FALLBACK = "VISUAL_ONLY_FALLBACK"


class FeatureUnavailable(ValueError):
    pass


def normalized_feature(value, dimension=None):
    """Return a unit ROI descriptor or fail; never manufacture a descriptor."""
    if value is None:
        raise FeatureUnavailable("FEATURE_UNAVAILABLE: GMT ROI descriptor missing")
    f = np.asarray(value, dtype=np.float32).reshape(-1)
    if dimension is not None and f.shape != (int(dimension),):
        raise FeatureUnavailable(f"FEATURE_DIMENSION_MISMATCH: {f.size} != manifest {dimension}")
    norm = float(np.linalg.norm(f))
    if not f.size or not np.isfinite(f).all() or not np.isfinite(norm) or norm < 1e-8:
        raise FeatureUnavailable("FEATURE_UNAVAILABLE: nonfinite or zero GMT ROI descriptor")
    return f / norm


@dataclass
class WorldObjectObservation:
    camera_id: int
    detection_id: int
    bbox_xywh: np.ndarray
    score: float
    roi_feature: np.ndarray | None
    feature_valid: bool
    world_center: np.ndarray
    world_covariance: np.ndarray
    spatial_support: object | None
    geometry_valid: bool
    geometry_reliability: float
    timestamp: int
    geometry_status: str = GEOMETRY_VALID
    visible: bool = True
    coordinate_frame: str = "PERSISTENT_WORLD"
    source: dict = field(default_factory=dict)
    point_support_available: bool | None = None
    cross_view_fusion_eligible: bool | None = None
    world_state_update_eligible: bool | None = None

    def __post_init__(self):
        self.camera_id = int(self.camera_id)
        self.detection_id = int(self.detection_id)
        self.timestamp = int(self.timestamp)
        if self.camera_id<0 or self.detection_id<0:
            raise ValueError("INVALID_CAMERA_OR_DETECTION_ID")
        self.bbox_xywh = np.asarray(self.bbox_xywh, dtype=np.float64).reshape(4).copy()
        self.score = float(self.score)
        if not np.isfinite(self.bbox_xywh).all() or np.any(self.bbox_xywh[2:] <= 0):
            raise ValueError("INVALID_CURRENT_DETECTION_BOX")
        if not np.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("INVALID_DETECTION_SCORE")
        if self.feature_valid:
            self.roi_feature = normalized_feature(self.roi_feature)
        else:
            self.roi_feature = None
        self.world_center = np.asarray(self.world_center, dtype=np.float64).reshape(3).copy()
        self.world_covariance = np.asarray(self.world_covariance, dtype=np.float64).reshape(3, 3).copy()
        self.geometry_reliability = float(self.geometry_reliability)
        if not np.isfinite(self.geometry_reliability):
            raise ValueError("NONFINITE_GEOMETRY_RELIABILITY")
        self.geometry_reliability = float(np.clip(self.geometry_reliability, 0, 1))
        cov = self.world_covariance
        valid = (self.coordinate_frame == "PERSISTENT_WORLD"
                 and np.isfinite(self.world_center).all() and np.isfinite(cov).all()
                 and np.allclose(cov, cov.T) and np.linalg.eigvalsh(cov).min() >= -1e-8
                 and self.geometry_reliability > 0)
        if self.geometry_valid and not valid:
            self.geometry_valid = False
            self.geometry_status = GEOMETRY_LOW_CONFIDENCE
        if not self.geometry_valid:
            self.geometry_reliability = 0.0
        support=self.spatial_support
        if self.point_support_available is None:
            self.point_support_available=bool(support is not None and support.point_support_available)
        strong=bool(self.geometry_valid and support is not None and support.cross_view_fusion_eligible)
        if self.cross_view_fusion_eligible is None:
            self.cross_view_fusion_eligible=strong
        if self.world_state_update_eligible is None:
            self.world_state_update_eligible=strong
        self.cross_view_fusion_eligible=bool(self.cross_view_fusion_eligible and strong)
        self.world_state_update_eligible=bool(self.world_state_update_eligible and strong)

    @property
    def key(self):
        return self.camera_id, self.detection_id

    @property
    def support(self):
        return self.spatial_support

    @property
    def feature(self):
        if not self.feature_valid:
            raise FeatureUnavailable("FEATURE_UNAVAILABLE: use geometry_only_ablation explicitly")
        return self.roi_feature


def fuse_identity_features(observations, dimension=None, use_geometry=True):
    """Only actual visible, valid ROI features contribute to long-term identity."""
    usable = [o for o in observations if o.visible and o.feature_valid and o.score > 0]
    if not usable:
        return None, 0.0
    features = [normalized_feature(o.roi_feature, dimension) for o in usable]
    if len({f.size for f in features}) != 1:
        raise FeatureUnavailable("FEATURE_DIMENSION_MISMATCH across cameras")
    # Geometry may fail without invalidating an actual visual observation.
    weights = np.array([o.score * (0.5 + 0.5 * o.geometry_reliability) if use_geometry else o.score for o in usable])
    fused = normalized_feature(np.average(np.stack(features), axis=0, weights=weights), dimension)
    return fused, float(np.average([o.score for o in usable], weights=weights))
