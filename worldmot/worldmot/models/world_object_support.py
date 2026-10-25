"""Causal, mask-free spatial support extraction for world objects.

The detector used by this experiment supplies boxes rather than instance
masks.  This module therefore keeps the distinction explicit: a rectangle is
an observation window, while the foreground support is a *proposal* inferred
from depth/point geometry.  It never consumes GT and returns an ambiguity flag
when depth alone cannot separate foreground from background.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class WorldObjectSupport:
    box_points: np.ndarray
    foreground_points: np.ndarray
    center: np.ndarray
    covariance: np.ndarray
    quality: float
    foreground_fraction: float
    ambiguity: str = "NONE"
    confidence_median: float = 0.0
    visible_fraction: float = 0.0
    diagnostics: dict = field(default_factory=dict)

    @property
    def reliable(self) -> bool:
        return bool(len(self.foreground_points) >= 3 and np.isfinite(self.center).all()
                    and self.quality > 0.0 and self.ambiguity != "NO_FINITE_SUPPORT")

    def to_dict(self, include_points: bool = True) -> dict:
        out = {
            "center": np.asarray(self.center, dtype=float).tolist(),
            "covariance": np.asarray(self.covariance, dtype=float).tolist(),
            "quality": float(self.quality),
            "foreground_fraction": float(self.foreground_fraction),
            "confidence_median": float(self.confidence_median),
            "visible_fraction": float(self.visible_fraction),
            "ambiguity": str(self.ambiguity),
            "diagnostics": dict(self.diagnostics),
            "box_point_count": int(len(self.box_points)),
            "foreground_point_count": int(len(self.foreground_points)),
        }
        if include_points:
            out["box_points"] = np.asarray(self.box_points, dtype=float).tolist()
            out["foreground_points"] = np.asarray(self.foreground_points, dtype=float).tolist()
        return out


def bbox_mask(bbox_xywh: Sequence[float], shape: tuple[int, int], margin: float = 0.0) -> np.ndarray:
    h, w = [int(x) for x in shape]
    x, y, bw, bh = [float(v) for v in bbox_xywh]
    x0 = max(0, int(np.floor(x + margin * bw)))
    y0 = max(0, int(np.floor(y + margin * bh)))
    x1 = min(w, int(np.ceil(x + (1.0 - margin) * bw)))
    y1 = min(h, int(np.ceil(y + (1.0 - margin) * bh)))
    out = np.zeros((h, w), dtype=bool)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = True
    return out


def map_bbox_to_model(bbox_xywh: Sequence[float], image_to_model: np.ndarray,
                      model_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Map an original-RGB ``xywh`` box into pointmap pixel coordinates.

    Ω predicts pointmaps after crop/resize/pad preprocessing. Detector boxes
    remain in the original RGB frame, so sampling a pointmap with the raw box
    silently selects the wrong pixels. Mapping all four corners keeps the
    registered affine preprocessing transform explicit and clips only at the
    pointmap boundary.
    """
    x, y, w, h = [float(v) for v in bbox_xywh]
    corners = np.asarray([[x, y, 1.0], [x + w, y, 1.0],
                          [x, y + h, 1.0], [x + w, y + h, 1.0]], dtype=np.float64)
    mapped = (np.asarray(image_to_model, dtype=np.float64) @ corners.T).T
    mapped = mapped[:, :2] / np.maximum(mapped[:, 2:3], 1e-12)
    lo = np.min(mapped, axis=0); hi = np.max(mapped, axis=0)
    if model_shape is not None:
        mh, mw = [int(v) for v in model_shape]
        lo = np.maximum(lo, 0.0); hi = np.minimum(hi, [mw, mh])
    return np.asarray([lo[0], lo[1], max(0.0, hi[0] - lo[0]), max(0.0, hi[1] - lo[1])], dtype=np.float64)


def _kmeans_depth(values: np.ndarray, iterations: int = 12) -> tuple[np.ndarray, float]:
    """Deterministic two-cluster 1-D k-means; returns labels and separation."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) < 8 or float(np.ptp(values)) <= 1e-8:
        return np.zeros(len(values), dtype=np.int8), 0.0
    centers = np.array([np.percentile(values, 25), np.percentile(values, 75)], dtype=np.float64)
    for _ in range(int(iterations)):
        labels = np.argmin(np.abs(values[:, None] - centers[None]), axis=1)
        new = np.array([values[labels == k].mean() if np.any(labels == k) else centers[k]
                        for k in range(2)])
        if np.allclose(new, centers, rtol=1e-5, atol=1e-7):
            break
        centers = new
    order = np.argsort(centers)
    remap = np.empty(2, dtype=np.int8)
    remap[order] = np.arange(2, dtype=np.int8)
    labels = remap[labels]
    separation = float(abs(centers[0] - centers[1]) / max(np.std(values), 1e-6))
    return labels, separation


def _trim(points: np.ndarray, max_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) <= int(max_points):
        return points
    idx = np.linspace(0, len(points) - 1, int(max_points), dtype=np.int64)
    return points[idx]


def _stats(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return np.full(3, np.nan), np.eye(3, dtype=np.float64) * 1e6
    lo, hi = np.percentile(points, [10, 90], axis=0)
    trimmed = points[np.all((points >= lo) & (points <= hi), axis=1)]
    if len(trimmed) < 3:
        trimmed = points
    center = np.median(trimmed, axis=0)
    cov = np.cov(trimmed.T) if len(trimmed) >= 3 else np.eye(3) * 1e-3
    return center.astype(np.float64), np.asarray(cov, dtype=np.float64).reshape(3, 3)


def extract_world_object_support(pointmap: np.ndarray, confidence: np.ndarray,
                                 bbox_xywh: Sequence[float], *, min_confidence: float = 1.5,
                                 max_points: int = 64, interior_margin: float = 0.05,
                                 use_foreground: bool = True) -> WorldObjectSupport:
    """Extract box and foreground supports without labels.

    The foreground hypothesis is selected from a compact depth cluster.  We
    intentionally abstain when the cluster is not separated or is too small;
    the caller can still retain ``box_points`` as a diagnostic/control.
    """
    pm = np.asarray(pointmap, dtype=np.float64)
    conf = np.asarray(confidence, dtype=np.float64)
    if pm.ndim != 3 or pm.shape[-1] != 3:
        raise ValueError(f"pointmap must be HxWx3, got {pm.shape}")
    if conf.shape != pm.shape[:2]:
        raise ValueError(f"confidence shape {conf.shape} != {pm.shape[:2]}")
    mask = bbox_mask(bbox_xywh, pm.shape[:2])
    inner = bbox_mask(bbox_xywh, pm.shape[:2], margin=float(interior_margin))
    finite = np.isfinite(pm).all(-1) & np.isfinite(conf) & (conf >= float(min_confidence))
    box = pm[mask & finite]
    candidate = pm[inner & finite]
    if len(candidate) == 0:
        center, cov = _stats(box)
        return WorldObjectSupport(_trim(box, max_points), np.empty((0, 3)), center, cov, 0.0,
                                  0.0, "NO_FINITE_SUPPORT", 0.0, 0.0,
                                  {"box_count": int(len(box)), "inner_count": 0})
    # Depth is camera-forward z when pointmaps come from Omega.  Norm is a
    # fallback for an alternate geometry backend with a nonstandard axis.
    depth = candidate[:, 2]
    if float(np.ptp(depth)) < 1e-6:
        depth = np.linalg.norm(candidate, axis=1)
    labels, separation = _kmeans_depth(depth)
    counts = np.bincount(labels, minlength=2)
    if not use_foreground or len(candidate) < 8:
        chosen = np.ones(len(candidate), dtype=bool)
        ambiguity = "BOX_ONLY" if not use_foreground else "LOW_SUPPORT"
    elif separation < 0.65 or counts.min() < 3:
        chosen = np.ones(len(candidate), dtype=bool)
        ambiguity = "DEPTH_NOT_SEPARATED"
    else:
        # The near cluster is the conservative person-surface hypothesis.  A
        # large far cluster is retained in diagnostics but does not silently
        # become foreground.
        chosen = labels == int(np.argmin([np.mean(depth[labels == k]) if np.any(labels == k) else np.inf
                                           for k in range(2)]))
        ambiguity = "DEPTH_CLUSTERED"
    foreground = candidate[chosen]
    center, cov = _stats(foreground)
    conf_vals = conf[inner & finite]
    visible_fraction = float(len(candidate) / max(1, int(inner.sum())))
    fg_fraction = float(len(foreground) / max(1, len(candidate)))
    quality = float(np.clip((len(foreground) / 16.0) * min(1.0, max(0.0, separation + 0.5))
                            * min(1.0, visible_fraction * 2.0), 0.0, 1.0))
    if ambiguity == "DEPTH_NOT_SEPARATED":
        quality *= 0.55
    return WorldObjectSupport(
        box_points=_trim(box, max_points), foreground_points=_trim(foreground, max_points),
        center=center, covariance=cov, quality=quality, foreground_fraction=fg_fraction,
        ambiguity=ambiguity, confidence_median=float(np.median(conf_vals)) if len(conf_vals) else 0.0,
        visible_fraction=visible_fraction,
        diagnostics={"box_count": int(len(box)), "inner_count": int(len(candidate)),
                     "foreground_count": int(len(foreground)), "depth_separation": separation,
                     "cluster_counts": counts.tolist(), "interior_margin": float(interior_margin)},
    )


def support_compatibility(left: WorldObjectSupport, right: WorldObjectSupport,
                          max_normalized_distance: float = 3.5) -> tuple[float, dict, str]:
    """Compare two supports while allowing different visible surfaces."""
    if not left.reliable or not right.reliable:
        return -np.inf, {}, "unreliable_support"
    delta = np.asarray(left.center) - np.asarray(right.center)
    scale = max(0.08, float(np.sqrt(np.trace(left.covariance + right.covariance) / 3.0)))
    normalized = float(np.linalg.norm(delta) / scale)
    cov = left.covariance + right.covariance + np.eye(3) * 1e-4
    mahal = float(np.sqrt(max(0.0, delta @ np.linalg.pinv(cov) @ delta)))
    ok = normalized <= float(max_normalized_distance) and mahal <= 6.0
    if not ok:
        # Do not spend an O(P^2) support-distance computation on pairs already
        # rejected by the cheap center/covariance gate.  The returned score
        # and reason are unchanged; ``support_nn_median`` is only meaningful
        # for geometrically admissible pairs.
        return -np.inf, {"normalized_distance": normalized, "mahalanobis": mahal}, "space_incompatible"
    # Symmetric nearest-neighbour distance is deliberately soft: opposite
    # cameras see different body surfaces, so exact support overlap is not
    # required.  It is evaluated only after the cheap gate above.
    a, b = left.foreground_points, right.foreground_points
    nn = max(float(np.median(np.min(np.linalg.norm(a[:, None] - b[None], axis=-1), axis=1))),
             float(np.median(np.min(np.linalg.norm(b[:, None] - a[None], axis=-1), axis=1))))
    score = float(np.exp(-0.5 * (normalized / 2.0) ** 2) * np.exp(-0.5 * (mahal / 4.0) ** 2)
                  * min(1.0, (left.quality + right.quality) / 2.0 + 0.25))
    return score, {"normalized_distance": normalized, "mahalanobis": mahal,
                   "support_nn_median": nn}, "accepted"

