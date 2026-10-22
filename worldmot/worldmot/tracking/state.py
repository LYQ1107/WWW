"""Persistent world-track state definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class TrackStatus(str, Enum):
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    LOST = "LOST"
    DELETED = "DELETED"


@dataclass
class WorldTrackState:
    global_id: int
    xyz_center: np.ndarray
    xyz_velocity: np.ndarray
    covariance: np.ndarray
    identity_feature: np.ndarray
    age: int = 1
    hits: int = 1
    missed: int = 0
    status: TrackStatus = TrackStatus.TENTATIVE
    last_timestamp: int = 0
    support_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    masks_by_camera: dict = field(default_factory=dict)
    source: str = "native-panoptic"

    def predict(self, dt: float = 1.0) -> None:
        self.xyz_center = self.xyz_center + self.xyz_velocity * float(dt)
        self.covariance = self.covariance + np.eye(3) * (0.02 * max(1.0, self.missed + 1))

    def update(self, center: np.ndarray, covariance: np.ndarray, feature: np.ndarray,
               support_points: np.ndarray, masks_by_camera: dict, timestamp: int,
               source: str = "native-panoptic") -> None:
        center = np.asarray(center, dtype=np.float64).reshape(3)
        old = self.xyz_center.copy()
        self.xyz_center = center
        self.xyz_velocity = 0.65 * self.xyz_velocity + 0.35 * (center - old)
        self.covariance = 0.65 * self.covariance + 0.35 * np.asarray(covariance, dtype=np.float64).reshape(3, 3)
        f = np.asarray(feature, dtype=np.float32).reshape(-1)
        if self.identity_feature.shape == f.shape and np.linalg.norm(f) > 1e-8:
            f = 0.8 * self.identity_feature + 0.2 * f
            f /= max(np.linalg.norm(f), 1e-8)
        self.identity_feature = f
        self.support_points = np.asarray(support_points, dtype=np.float64).reshape(-1, 3)
        self.masks_by_camera = masks_by_camera
        self.last_timestamp = int(timestamp)
        self.age += 1
        self.hits += 1
        self.missed = 0
        if self.status in (TrackStatus.TENTATIVE, TrackStatus.LOST) and self.hits >= 2:
            self.status = TrackStatus.CONFIRMED
        self.source = source

    def mark_missed(self, timestamp: int, max_missed: int) -> None:
        self.last_timestamp = int(timestamp)
        self.age += 1
        self.missed += 1
        if self.missed > max_missed:
            self.status = TrackStatus.DELETED
        elif self.status == TrackStatus.CONFIRMED:
            self.status = TrackStatus.LOST

    def to_dict(self) -> dict:
        return {
            "global_id": int(self.global_id), "xyz_center": self.xyz_center.tolist(),
            "xyz_velocity": self.xyz_velocity.tolist(), "covariance": self.covariance.tolist(),
            "age": int(self.age), "hits": int(self.hits), "missed": int(self.missed),
            "status": self.status.value, "last_timestamp": int(self.last_timestamp),
            "source": self.source,
        }
