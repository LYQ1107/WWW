"""Robust updates for persistent world-object state."""

from __future__ import annotations

import numpy as np


def update_world_measurement(center: np.ndarray, covariance: np.ndarray,
                             measurement: np.ndarray, measurement_covariance: np.ndarray,
                             reliability: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Information-form update with reliability clamped to [0, 1]."""
    r = float(np.clip(reliability, 0.0, 1.0))
    c = np.asarray(covariance, dtype=np.float64).reshape(3, 3)
    m = np.asarray(measurement, dtype=np.float64).reshape(3)
    mc = np.asarray(measurement_covariance, dtype=np.float64).reshape(3, 3)
    if (r <= 0 or not np.isfinite(m).all() or not np.isfinite(mc).all()
            or not np.allclose(mc, mc.T) or np.linalg.eigvalsh(mc).min() < -1e-8):
        return np.asarray(center, dtype=np.float64).reshape(3).copy(), c.copy()
    mc = mc / max(r, 1e-3)
    try:
        p = np.linalg.pinv(c); q = np.linalg.pinv(mc)
        out_cov = np.linalg.pinv(p + q)
        out_center = out_cov @ (p @ np.asarray(center, dtype=np.float64).reshape(3) + q @ m)
        return out_center, out_cov
    except np.linalg.LinAlgError:
        return np.asarray(center, dtype=np.float64).reshape(3), c
