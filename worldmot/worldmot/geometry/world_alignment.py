"""Causal static-background Sim(3) alignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Sim3:
    scale: float = 1.0
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def __post_init__(self) -> None:
        self.scale = float(self.scale)
        self.rotation = np.asarray(self.rotation, dtype=np.float64).reshape(3, 3)
        self.translation = np.asarray(self.translation, dtype=np.float64).reshape(3)

    def apply(self, points: np.ndarray) -> np.ndarray:
        x = np.asarray(points, dtype=np.float64)
        return self.scale * (x @ self.rotation.T) + self.translation

    def matrix(self) -> np.ndarray:
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = self.scale * self.rotation
        out[:3, 3] = self.translation
        return out


@dataclass
class AlignmentResult:
    success: bool
    transform: Sim3
    correspondences: int
    inliers: int
    inlier_ratio: float
    residual_median: float
    residual_p95: float
    reason: str = ""
    diagnostics: dict = field(default_factory=dict)


def _umeyama(src: np.ndarray, dst: np.ndarray, estimate_scale: bool = True) -> Sim3:
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if len(src) < 3 or len(dst) != len(src):
        raise ValueError("Sim(3) needs at least three paired points")
    mu_src, mu_dst = src.mean(0), dst.mean(0)
    xs, xd = src - mu_src, dst - mu_dst
    cov = (xd.T @ xs) / len(src)
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1] = -1
    r = u @ np.diag(d) @ vt
    var_src = np.mean(np.sum(xs * xs, axis=1))
    scale = float((s * d).sum() / max(var_src, 1e-12)) if estimate_scale else 1.0
    t = mu_dst - scale * (r @ mu_src)
    return Sim3(scale, r, t)


def estimate_sim3(src: np.ndarray, dst: np.ndarray, threshold: float = 0.08,
                  min_inliers: int = 6, iterations: int = 256,
                  seed: int = 0, estimate_scale: bool = True) -> AlignmentResult:
    """Robustly estimate ``dst ~= T(src)`` with deterministic RANSAC."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(src).all(1) & np.isfinite(dst).all(1)
    src, dst = src[finite], dst[finite]
    n = len(src)
    identity = Sim3()
    if n < 3:
        return AlignmentResult(False, identity, n, 0, 0.0, float("inf"), float("inf"), "too_few_correspondences")
    rng = np.random.default_rng(seed)
    best_mask = None
    best_score = (-1, float("inf"))
    sample_size = 3 if estimate_scale else 3
    for _ in range(max(1, int(iterations))):
        idx = rng.choice(n, sample_size, replace=False)
        try:
            candidate = _umeyama(src[idx], dst[idx], estimate_scale=estimate_scale)
        except (np.linalg.LinAlgError, ValueError):
            continue
        residual = np.linalg.norm(candidate.apply(src) - dst, axis=1)
        mask = residual <= threshold
        score = (int(mask.sum()), float(np.median(residual[mask])) if mask.any() else float("inf"))
        if score[0] > best_score[0] or (score[0] == best_score[0] and score[1] < best_score[1]):
            best_score, best_mask = score, mask
    if best_mask is None or int(best_mask.sum()) < max(3, int(min_inliers)):
        return AlignmentResult(False, identity, n, int(best_mask.sum()) if best_mask is not None else 0,
                               (int(best_mask.sum()) / n) if best_mask is not None else 0.0,
                               float("inf"), float("inf"), "insufficient_inliers")
    try:
        transform = _umeyama(src[best_mask], dst[best_mask], estimate_scale=estimate_scale)
    except (np.linalg.LinAlgError, ValueError):
        return AlignmentResult(False, identity, n, int(best_mask.sum()), float(best_mask.mean()),
                               float("inf"), float("inf"), "refit_failed")
    residual = np.linalg.norm(transform.apply(src) - dst, axis=1)
    inlier = residual <= threshold
    vals = residual[inlier]
    return AlignmentResult(True, transform, n, int(inlier.sum()), float(inlier.mean()),
                           float(np.median(vals)), float(np.percentile(vals, 95)), "ok",
                           {"threshold": float(threshold), "scale": float(transform.scale)})


class Sim3Aligner:
    def __init__(self, threshold: float = 0.08, min_inliers: int = 6, seed: int = 0):
        self.threshold = float(threshold)
        self.min_inliers = int(min_inliers)
        self.seed = int(seed)
        self.world_from_current = Sim3()
        self.initialized = False

    def reset(self) -> None:
        self.world_from_current = Sim3()
        self.initialized = False

    def initialize(self) -> AlignmentResult:
        self.world_from_current = Sim3()
        self.initialized = True
        return AlignmentResult(True, self.world_from_current, 0, 0, 1.0, 0.0, 0.0, "initial_reference")

    def update(self, current_points: np.ndarray, previous_world_points: np.ndarray) -> AlignmentResult:
        result = estimate_sim3(current_points, previous_world_points,
                                threshold=self.threshold, min_inliers=self.min_inliers, seed=self.seed)
        if result.success:
            self.world_from_current = result.transform
            self.initialized = True
        return result
