"""Camera-conditioned query projection utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QueryProjection:
    camera_id: int
    global_id: int
    uv: np.ndarray | None
    visible: bool
    evaluation_eligible: bool
    reason: str


def project_world_query(camera, xyz_world: np.ndarray, camera_id: int, global_id: int,
                        has_current_observation: bool = False,
                        coordinate_system_valid: bool = False) -> QueryProjection:
    uv, valid = camera.project(np.asarray(xyz_world, dtype=np.float64).reshape(1, 3), clip=False)
    if not bool(valid[0]):
        return QueryProjection(int(camera_id), int(global_id), None, False, False, "behind_camera_or_invalid")
    h, w = camera.original_size
    inside = 0 <= float(uv[0, 0]) < w and 0 <= float(uv[0, 1]) < h
    eligible = bool(inside and has_current_observation and coordinate_system_valid)
    if not coordinate_system_valid:
        reason = "clip_normalized_query_projection_only"
    elif has_current_observation:
        reason = "current_observation"
    else:
        reason = "projection_only"
    return QueryProjection(int(camera_id), int(global_id), uv[0].astype(float), bool(inside), eligible, reason)
