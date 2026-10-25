"""Explicit Ω-local-world -> persistent-world Sim(3) contract."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .world_alignment import Sim3


@dataclass(frozen=True)
class WorldFrameTransform:
    transform: Sim3
    timestamp: int
    valid: bool
    source_frame: str = "OMEGA_LOCAL_WORLD"
    target_frame: str = "PERSISTENT_WORLD"

    def __post_init__(self):
        t = self.transform
        if self.valid and (not np.isfinite(t.matrix()).all() or t.scale <= 0
                           or not np.allclose(t.rotation.T @ t.rotation, np.eye(3), atol=1e-6)
                           or not np.isclose(np.linalg.det(t.rotation), 1.0, atol=1e-6)):
            raise ValueError("INVALID_SIM3")

    def apply(self, points):
        if not self.valid:
            raise ValueError("WORLD_ALIGNMENT_FAILED: cannot reuse historical transform")
        return self.transform.apply(points)

    def covariance(self, covariance):
        if not self.valid:
            raise ValueError("WORLD_ALIGNMENT_FAILED")
        t = self.transform
        return t.scale ** 2 * t.rotation @ np.asarray(covariance) @ t.rotation.T

    def support(self, support):
        from dataclasses import replace
        if support.coordinate_frame != self.source_frame:
            raise ValueError("SUPPORT_FRAME_MISMATCH")
        return replace(support, box_points=self.apply(support.box_points),
                       foreground_points=self.apply(support.foreground_points),
                       center=self.apply(support.center[None])[0],
                       covariance=self.covariance(support.covariance),
                       coordinate_frame=self.target_frame)
