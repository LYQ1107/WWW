"""Geometry features used by the persistent-world-query model.

The Omega reconstruction is uncalibrated and its scale is only locally
meaningful.  This module therefore encodes normalized relative geometry and
keeps the scale used for a clip in the returned metadata.  No fixed value is
interpreted as metres.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class GeometryNormalization:
    """Robust per-clip scale used for world-query features."""

    origin: np.ndarray
    scale: float
    source: str = "observation_median_mad"

    def normalize(self, xyz: np.ndarray) -> np.ndarray:
        return (np.asarray(xyz, dtype=np.float32) - self.origin) / max(float(self.scale), 1e-6)


def fit_geometry_normalization(points: np.ndarray) -> GeometryNormalization:
    """Fit a deterministic robust scale from observed (non-GT) positions."""

    x = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    finite = x[np.isfinite(x).all(axis=1)]
    if len(finite) == 0:
        return GeometryNormalization(np.zeros(3, dtype=np.float32), 1.0, "empty_fallback")
    origin = np.median(finite, axis=0)
    radius = np.linalg.norm(finite - origin[None], axis=1)
    scale = float(np.median(radius) + 1.4826 * np.median(np.abs(radius - np.median(radius))))
    if not np.isfinite(scale) or scale < 1e-3:
        scale = float(np.percentile(radius, 75)) if len(radius) else 1.0
    return GeometryNormalization(origin.astype(np.float32), max(scale, 1e-3))


def encode_relative_geometry(
    xyz: np.ndarray,
    velocity: np.ndarray | None = None,
    covariance: np.ndarray | None = None,
    camera_pose: np.ndarray | None = None,
    geometry_reliability: float = 1.0,
    visible: float = 1.0,
    normalization: GeometryNormalization | None = None,
) -> np.ndarray:
    """Return a fixed, non-learned geometry vector for one observation/query.

    The vector contains normalized position, velocity, log uncertainty, a
    compact camera pose, and explicit reliability/visibility flags.  The
    camera pose is optional because Omega can fail to produce a valid pose.
    """

    xyz = np.asarray(xyz, dtype=np.float32).reshape(3)
    normal = normalization or fit_geometry_normalization(xyz[None])
    pos = normal.normalize(xyz)
    vel = np.zeros(3, dtype=np.float32) if velocity is None else np.asarray(velocity, dtype=np.float32).reshape(3) / max(normal.scale, 1e-6)
    cov = np.eye(3, dtype=np.float32) if covariance is None else np.asarray(covariance, dtype=np.float32).reshape(3, 3)
    cov_diag = np.log1p(np.clip(np.diag(cov), 0.0, 1e6)).astype(np.float32)
    pose = np.zeros(6, dtype=np.float32)
    if camera_pose is not None:
        p = np.asarray(camera_pose, dtype=np.float32)
        if p.shape == (4, 4):
            pose[:3] = p[:3, 3] / max(normal.scale, 1e-6)
            # A compact rotation representation that is continuous near the
            # identity; full matrices are retained in the observation record.
            pose[3:] = np.asarray([p[2, 1] - p[1, 2], p[0, 2] - p[2, 0], p[1, 0] - p[0, 1]], dtype=np.float32) / 2.0
    return np.concatenate([
        pos, vel, cov_diag, pose,
        np.asarray([float(geometry_reliability), float(visible), float(normal.scale)], dtype=np.float32),
    ]).astype(np.float32)


def encode_batch_geometry(records: list[Mapping], normalization: GeometryNormalization | None = None) -> np.ndarray:
    """Encode a list of observation dictionaries without consulting labels."""

    points = np.asarray([r.get("xyz_world", r.get("xyz", [np.nan] * 3)) for r in records], dtype=np.float32)
    normal = normalization or fit_geometry_normalization(points)
    out = []
    for r in records:
        out.append(encode_relative_geometry(
            r.get("xyz_world", r.get("xyz", [np.nan] * 3)),
            r.get("velocity"), r.get("covariance"), r.get("camera_pose"),
            r.get("geometry_reliability", 0.0), r.get("visible", 1.0), normal,
        ))
    return np.stack(out, axis=0) if out else np.empty((0, 18), dtype=np.float32)
