"""Backend contract shared by PanSt3R and future geometry implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..geometry.camera_model import CameraModel


@dataclass
class GeometryFrame:
    timestamp: int
    images: Mapping[int, np.ndarray]
    pointmaps: Mapping[int, np.ndarray]
    confidences: Mapping[int, np.ndarray]
    panoptic: Mapping[int, dict]
    query_features: np.ndarray
    cameras: Mapping[int, CameraModel]
    image_to_model: Mapping[int, np.ndarray] = field(default_factory=dict)
    model_to_image: Mapping[int, np.ndarray] = field(default_factory=dict)
    # Detector masks are kept in original-image coordinates.  They are used
    # only for causal static-background filtering and for lifting detections;
    # they are never populated from ground truth.
    dynamic_masks: Mapping[int, np.ndarray] = field(default_factory=dict)
    detector_outputs: Mapping[int, list] = field(default_factory=dict)
    status: str = "OK"
    diagnostics: dict = field(default_factory=dict)

    def finite(self) -> bool:
        return all(np.isfinite(np.asarray(x)).all() for x in self.pointmaps.values())


class GeometryBackend(ABC):
    """Geometry backend with full reconstruction and future update hooks."""

    @abstractmethod
    def reconstruct(self, images: Mapping[int, np.ndarray], timestamp: int) -> GeometryFrame:
        raise NotImplementedError

    def update(self, images: Mapping[int, np.ndarray], previous_state: Any, timestamp: int) -> GeometryFrame:
        return self.reconstruct(images, timestamp)
