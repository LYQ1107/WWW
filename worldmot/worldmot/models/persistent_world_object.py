"""Data contract for the learned Persistent Multi-View World Object tracker."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class WorldObjectState:
    geometry_state: np.ndarray
    spatial_support: np.ndarray
    uncertainty: np.ndarray
    memory_embedding: np.ndarray
    visibility_by_camera: dict[int, bool] = field(default_factory=dict)
    life_state: str = "TENTATIVE"

    def as_vector(self) -> np.ndarray:
        return np.concatenate([self.geometry_state.reshape(-1), self.uncertainty.reshape(-1),
                               self.memory_embedding.reshape(-1)]).astype(np.float32)


@dataclass
class PersistentWorldObjectObservation:
    camera_id: int
    center: np.ndarray
    covariance: np.ndarray
    support_points: np.ndarray
    visual_feature: np.ndarray
    reliability: float
    detection_id: int

