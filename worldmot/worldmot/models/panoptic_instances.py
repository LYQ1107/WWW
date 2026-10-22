"""Convert one joint PanSt3R panoptic output into shared 3-D proposals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional

import numpy as np


@dataclass
class WorldInstance:
    proposal_id: int
    query_feature: np.ndarray
    xyz_center: np.ndarray
    xyz_support: np.ndarray
    covariance: np.ndarray
    confidence: float
    visible_cameras: tuple[int, ...]
    masks_by_camera: Dict[int, np.ndarray] = field(default_factory=dict)
    geometry_valid: bool = True
    source: str = "native-panoptic"
    category_id: int | None = None

    def transformed(self, transform) -> "WorldInstance":
        # Covariance lives in the proposal's coordinate frame.  Carry it into
        # the persistent world frame together with the support points so the
        # identity gate does not silently mix uncertainties from two frames.
        linear = float(transform.scale) * np.asarray(transform.rotation, dtype=np.float64)
        covariance = linear @ self.covariance @ linear.T
        return WorldInstance(
            proposal_id=self.proposal_id,
            query_feature=self.query_feature.copy(),
            xyz_center=transform.apply(self.xyz_center[None])[0],
            xyz_support=transform.apply(self.xyz_support),
            covariance=covariance,
            confidence=self.confidence,
            visible_cameras=self.visible_cameras,
            masks_by_camera=self.masks_by_camera,
            geometry_valid=self.geometry_valid,
            source=self.source,
            category_id=self.category_id,
        )


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _normalize_pointmap(x) -> np.ndarray:
    x = _to_numpy(x)
    while x.ndim > 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"pointmap must be HxWx3, got {x.shape}")
    if x.shape[-1] != 3 and x.shape[0] == 3:
        x = np.moveaxis(x, 0, -1)
    if x.shape[-1] != 3:
        raise ValueError(f"pointmap last dimension must be 3, got {x.shape}")
    return x.astype(np.float64, copy=False)


def _normalize_conf(x, shape) -> np.ndarray:
    c = np.ones(shape, dtype=np.float64) if x is None else _to_numpy(x)
    while c.ndim > 2 and c.shape[0] == 1:
        c = c[0]
    if c.ndim == 3:
        c = c[0] if c.shape[0] == 1 else c.mean(-1)
    if c.shape != shape:
        # Confidence is auxiliary; nearest resize keeps the conservative mask.
        from PIL import Image
        c = np.asarray(Image.fromarray(c.astype(np.float32), mode="F").resize((shape[1], shape[0]), Image.Resampling.NEAREST))
    return c.astype(np.float64, copy=False)


def _mask_for_segment(pan, segment_id, shape) -> np.ndarray:
    p = _to_numpy(pan)
    while p.ndim > 2 and p.shape[0] == 1:
        p = p[0]
    if p.shape != shape:
        from PIL import Image
        p = np.asarray(Image.fromarray(p.astype(np.int32), mode="I").resize((shape[1], shape[0]), Image.Resampling.NEAREST))
    return p == int(segment_id)


def _feature(query_features: np.ndarray, query_id: int) -> np.ndarray:
    f = np.asarray(query_features)
    if f.ndim == 3:
        f = f[0]
    if f.ndim != 2 or query_id < 0 or query_id >= len(f):
        return np.zeros(1, dtype=np.float32)
    out = f[query_id].astype(np.float32, copy=True)
    norm = np.linalg.norm(out)
    return out / norm if norm > 1e-8 else out


def build_world_instances(pointmaps: Mapping[int, np.ndarray], confidences: Mapping[int, np.ndarray],
                          panoptic: Mapping[int, dict], query_features: np.ndarray,
                          min_confidence: float = 1.5, min_points: int = 24,
                          max_support: int = 2048, person_category_ids: Optional[Iterable[int]] = None) -> list[WorldInstance]:
    """Build one proposal per shared query id.

    PanSt3R's query id is deliberately scoped to this synchronized forward;
    this function never turns it into a persistent track id.
    """
    allowed = None if person_category_ids is None else {int(x) for x in person_category_ids}
    by_query: Dict[int, dict] = {}
    for view, pred in panoptic.items():
        pm = _normalize_pointmap(pointmaps[view])
        conf = _normalize_conf(confidences.get(view), pm.shape[:2])
        pan = pred.get("pan") if isinstance(pred, dict) else None
        segments = pred.get("segments_info", []) if isinstance(pred, dict) else []
        if isinstance(pan, (list, tuple)):
            # A multi-view batch is allowed; this adapter receives one view at a time.
            pan = pan[0] if len(pan) == 1 else pan
        for seg in segments:
            category_id = seg.get("category_id")
            category_name = str(seg.get("category_name", ""))
            if category_name and category_name.lower() != "person":
                continue
            if allowed is not None and category_id is not None and int(category_id) not in allowed:
                continue
            if "query_id" not in seg or pan is None:
                continue
            q = int(seg["query_id"])
            mask = _mask_for_segment(pan, seg.get("id", q + 1), pm.shape[:2])
            good = mask & np.isfinite(pm).all(-1) & np.isfinite(conf) & (conf >= float(min_confidence))
            pts = pm[good]
            if len(pts) < int(min_points):
                continue
            # Deterministic spatial subsampling avoids retaining full dense maps.
            if len(pts) > max_support:
                idx = np.linspace(0, len(pts) - 1, max_support, dtype=np.int64)
                pts = pts[idx]
            entry = by_query.setdefault(q, {"points": [], "masks": {}, "conf": [], "category_id": category_id})
            entry["points"].append(pts)
            entry["masks"][int(view)] = mask
            entry["conf"].append(float(np.median(conf[good])))
            if entry.get("category_id") is None:
                entry["category_id"] = category_id
    out: list[WorldInstance] = []
    for q in sorted(by_query):
        entry = by_query[q]
        support = np.concatenate(entry["points"], axis=0)
        # Coordinate-wise trimmed center is less sensitive to background leakage.
        lo, hi = np.percentile(support, [10, 90], axis=0)
        trimmed = support[np.all((support >= lo) & (support <= hi), axis=1)]
        if len(trimmed) < 3:
            trimmed = support
        center = np.median(trimmed, axis=0)
        cov = np.cov(trimmed.T) if len(trimmed) >= 3 else np.eye(3) * 1e-3
        cov = np.asarray(cov, dtype=np.float64).reshape(3, 3)
        out.append(WorldInstance(
            proposal_id=q, query_feature=_feature(query_features, q),
            xyz_center=center.astype(np.float64), xyz_support=support.astype(np.float64),
            covariance=cov, confidence=float(np.mean(entry["conf"])),
            visible_cameras=tuple(sorted(entry["masks"])), masks_by_camera=entry["masks"],
            geometry_valid=bool(np.isfinite(center).all()), category_id=entry.get("category_id"),
        ))
    return out
