"""Pure 3-D to 2-D projection used by the online runtime and tests."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .camera_model import CameraModel


@dataclass
class ProjectionResult:
    visible: bool
    bbox_xywh: tuple[float, float, float, float] | None
    uv: np.ndarray
    valid: np.ndarray
    visible_fraction: float
    # Projection only proves that world support lands in front of and inside
    # the camera.  It does not prove a target-view RGB observation or absence
    # of occlusion; callers must combine these fields with current masks.
    geometry_visible: bool = False
    occlusion_unknown: bool = True
    visibility_evidence: str = "projection_only"


def project_support(points_world: np.ndarray, camera: CameraModel, min_points: int = 3,
                    clip: bool = True) -> ProjectionResult:
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return ProjectionResult(False, None, np.empty((0, 2)), np.empty((0,), dtype=bool), 0.0,
                                geometry_visible=False)
    uv, valid = camera.project(points, clip=False)
    in_frame = valid.copy()
    h, w = camera.original_size
    in_frame &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    good = uv[in_frame]
    if len(good) < int(min_points):
        fraction = float(len(good) / max(1, len(points)))
        return ProjectionResult(False, None, uv, in_frame, fraction,
                                geometry_visible=False)
    x0, y0 = good.min(0)
    x1, y1 = good.max(0)
    if clip:
        x0, x1 = np.clip([x0, x1], 0, w - 1)
        y0, y1 = np.clip([y0, y1], 0, h - 1)
    bbox = (float(x0), float(y0), float(max(0.0, x1 - x0)), float(max(0.0, y1 - y0)))
    return ProjectionResult(True, bbox, uv, in_frame, float(len(good) / max(1, len(points))),
                           geometry_visible=True)
