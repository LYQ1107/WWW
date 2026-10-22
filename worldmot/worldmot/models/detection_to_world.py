"""Lift 2-D person detections into Omega's shared 3-D frame and fuse views."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np

from .person_detector import PersonDetection
from .omega_geometry_backend import PixelMapping, map_mask_to_model
from ..geometry.reprojection import project_support
from ..models.panoptic_instances import WorldInstance


@dataclass
class WorldDetectionObservation:
    observation_id: str
    detections: list[PersonDetection]
    support_points: np.ndarray
    center: np.ndarray
    covariance: np.ndarray
    confidence: float
    geometry_reliable: bool
    fusion_status: str
    support_points_by_camera: dict[int, np.ndarray] = field(default_factory=dict)
    rejection_reason: str = ""
    reprojection_coverage: dict[str, float] = field(default_factory=dict)

    @property
    def cameras(self) -> tuple[int, ...]:
        return tuple(sorted({int(x.view_id) for x in self.detections}))

    def to_world_instance(self, proposal_id: int) -> WorldInstance:
        masks = {int(x.view_id): np.asarray(x.mask).astype(bool)
                 for x in self.detections if x.mask is not None}
        feature = np.asarray([self.confidence, float(len(self.detections))], dtype=np.float32)
        feature /= max(np.linalg.norm(feature), 1e-8)
        return WorldInstance(
            proposal_id=int(proposal_id), query_feature=feature,
            xyz_center=np.asarray(self.center, dtype=np.float64),
            xyz_support=np.asarray(self.support_points, dtype=np.float64),
            covariance=np.asarray(self.covariance, dtype=np.float64),
            confidence=float(self.confidence), visible_cameras=self.cameras,
            masks_by_camera=masks,
            support_points_by_camera={int(v): np.asarray(p, dtype=np.float64)
                                     for v, p in self.support_points_by_camera.items()},
            geometry_valid=bool(self.geometry_reliable),
            source="rtmdet-ins-tiny+omega512", category_id=0,
        )

    def to_dict(self, include_support: bool = True) -> dict:
        out = {
            "observation_id": self.observation_id,
            "camera_ids": list(self.cameras),
            "detection_ids": [f"{x.view_id}:{x.detection_id}" for x in self.detections],
            "detections": [x.to_dict(include_mask=False) for x in self.detections],
            "center": np.asarray(self.center, dtype=float).tolist(),
            "covariance": np.asarray(self.covariance, dtype=float).tolist(),
            "confidence": float(self.confidence),
            "geometry_reliable": bool(self.geometry_reliable),
            "fusion_status": self.fusion_status,
            "rejection_reason": self.rejection_reason,
            "reprojection_coverage": {str(k): float(v) for k, v in self.reprojection_coverage.items()},
        }
        if include_support:
            out["support_points"] = np.asarray(self.support_points, dtype=float).tolist()
            out["support_points_by_camera"] = {
                str(int(v)): np.asarray(p, dtype=float).tolist()
                for v, p in self.support_points_by_camera.items()
            }
        return out


def _mapping_from_frame(frame, view: int) -> PixelMapping:
    record = (frame.diagnostics.get("mappings", {}) or {}).get(str(int(view)))
    if record:
        values = {k: record[k] for k in (
            "original_width", "original_height", "crop_left", "crop_top",
            "crop_width", "crop_height", "resize_width", "resize_height",
            "pad_left", "pad_top", "network_width", "network_height")}
        return PixelMapping(**{k: int(v) for k, v in values.items()})
    h, w = np.asarray(frame.images[view]).shape[:2]
    hm, wm = np.asarray(frame.pointmaps[view]).shape[:2]
    return PixelMapping(w, h, 0, 0, w, h, wm, hm, 0, 0, wm, hm)


def _mask_for_detection(det: PersonDetection, shape: tuple[int, int]) -> np.ndarray:
    if det.mask is not None:
        mask = np.asarray(det.mask).astype(bool)
        if mask.shape == shape:
            return mask
    x, y, w, h = det.bbox_xywh
    x0, y0 = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
    x1, y1 = min(shape[1], int(np.ceil(x + w))), min(shape[0], int(np.ceil(y + h)))
    mask = np.zeros(shape, dtype=bool)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


def lift_detection(det: PersonDetection, frame, min_confidence: float = 1.5,
                   min_points: int = 16, max_support: int = 512) -> WorldDetectionObservation:
    """Sample only the current Omega pointmap under the current 2-D mask."""
    view = int(det.view_id)
    points = np.asarray(frame.pointmaps[view], dtype=np.float64)
    conf = np.asarray(frame.confidences[view], dtype=np.float64)
    mapping = _mapping_from_frame(frame, view)
    mask_original = _mask_for_detection(det, np.asarray(frame.images[view]).shape[:2])
    mask_model = map_mask_to_model(mask_original, mapping)
    if mask_model.shape != points.shape[:2]:
        # This is only a defensive path for non-Omega backends.
        from PIL import Image
        mask_model = np.asarray(Image.fromarray(mask_model.astype(np.uint8)).resize(
            (points.shape[1], points.shape[0]), Image.Resampling.NEAREST)) > 0
    good = mask_model & np.isfinite(points).all(-1) & np.isfinite(conf) & (conf >= float(min_confidence))
    support = points[good]
    reliable = len(support) >= int(min_points)
    reason = "" if reliable else f"insufficient_geometry_support:{len(support)}<{int(min_points)}"
    if len(support) == 0:
        center = np.full(3, np.nan, dtype=np.float64)
        covariance = np.eye(3, dtype=np.float64) * 1e6
    else:
        if len(support) > int(max_support):
            support = support[np.linspace(0, len(support) - 1, int(max_support), dtype=np.int64)]
        lo, hi = np.percentile(support, [10, 90], axis=0)
        trimmed = support[np.all((support >= lo) & (support <= hi), axis=1)]
        if len(trimmed) < 3:
            trimmed = support
        center = np.median(trimmed, axis=0)
        covariance = np.cov(trimmed.T) if len(trimmed) >= 3 else np.eye(3) * 1e-3
        covariance = np.asarray(covariance, dtype=np.float64).reshape(3, 3)
    # Retain the original mask on the detection, because WorldMOT's static
    # mask filter operates in original RGB coordinates.
    det.mask = mask_original
    return WorldDetectionObservation(
        observation_id=f"{view}:{det.detection_id}", detections=[det],
        support_points=np.asarray(support, dtype=np.float64), center=center,
        covariance=covariance, confidence=float(det.score),
        geometry_reliable=bool(reliable), fusion_status="SINGLE_VIEW" if reliable else "REJECTED",
        support_points_by_camera={view: np.asarray(support, dtype=np.float64)},
        rejection_reason=reason,
    )


def _mask_coverage(points: np.ndarray, target: PersonDetection, camera) -> float:
    if len(points) == 0:
        return 0.0
    projected = camera.project(points, clip=False)
    uv, valid = projected
    h, w = camera.original_size
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    if not valid.any():
        return 0.0
    if target.mask is not None and np.asarray(target.mask).shape == (h, w):
        xy = np.rint(uv[valid]).astype(np.int64)
        xy[:, 0] = np.clip(xy[:, 0], 0, w - 1); xy[:, 1] = np.clip(xy[:, 1], 0, h - 1)
        return float(np.mean(np.asarray(target.mask)[xy[:, 1], xy[:, 0]]))
    x, y, bw, bh = target.bbox_xywh
    return float(np.mean((uv[valid, 0] >= x) & (uv[valid, 0] <= x + bw) &
                         (uv[valid, 1] >= y) & (uv[valid, 1] <= y + bh)))


def _compatibility(left: WorldDetectionObservation, right: WorldDetectionObservation,
                   cameras: Mapping[int, object]) -> tuple[float, dict[str, float], str]:
    if not left.geometry_reliable or not right.geometry_reliable:
        return -np.inf, {}, "unreliable_geometry"
    if left.detections[0].view_id == right.detections[0].view_id:
        return -np.inf, {}, "same_camera"
    center_distance = float(np.linalg.norm(left.center - right.center))
    scale_l = float(np.percentile(np.linalg.norm(left.support_points - left.center, axis=1), 80))
    scale_r = float(np.percentile(np.linalg.norm(right.support_points - right.center, axis=1), 80))
    scale = max(0.08, scale_l, scale_r)
    normalized_distance = center_distance / scale
    cov = left.covariance + right.covariance + np.eye(3) * max(1e-4, scale * scale * 0.02)
    try:
        mahalanobis = float(np.sqrt((left.center - right.center) @ np.linalg.pinv(cov) @ (left.center - right.center)))
    except Exception:
        mahalanobis = float("inf")
    lv, rv = left.detections[0].view_id, right.detections[0].view_id
    coverage_l_to_r = _mask_coverage(left.support_points, right.detections[0], cameras[rv])
    coverage_r_to_l = _mask_coverage(right.support_points, left.detections[0], cameras[lv])
    coverage = min(coverage_l_to_r, coverage_r_to_l)
    # Surface points from opposite views need not overlap.  A close, low
    # uncertainty pair is accepted with weak coverage; large uncertainty or
    # a large centre mismatch is abstained rather than merged by nearest
    # centroid alone.
    accepted = (mahalanobis <= 5.0 and normalized_distance <= 3.5 and
                (coverage >= 0.03 or normalized_distance <= 1.2))
    if not accepted:
        return -np.inf, {"center_distance": center_distance, "normalized_distance": normalized_distance,
                         "mahalanobis": mahalanobis, "coverage_left_to_right": coverage_l_to_r,
                         "coverage_right_to_left": coverage_r_to_l}, "geometry_incompatible"
    score = (0.45 * np.exp(-0.5 * min(10.0, mahalanobis) ** 2 / 4.0) +
             0.35 * np.exp(-0.5 * min(10.0, normalized_distance) ** 2 / 2.0) +
             0.20 * coverage)
    return float(score), {"center_distance": center_distance, "normalized_distance": normalized_distance,
                          "mahalanobis": mahalanobis, "coverage_left_to_right": coverage_l_to_r,
                          "coverage_right_to_left": coverage_r_to_l}, "accepted"


def fuse_two_views(observations: Iterable[WorldDetectionObservation], frame) -> tuple[list[WorldDetectionObservation], dict]:
    """Fuse at most one observation per camera using geometry + reprojection gates."""
    obs = [x for x in observations]
    by_view: dict[int, list[WorldDetectionObservation]] = {}
    for x in obs:
        by_view.setdefault(int(x.detections[0].view_id), []).append(x)
    views = sorted(by_view)
    if len(views) < 2:
        return obs, {"candidate_pairs": 0, "accepted_pairs": 0, "rejected_pairs": 0, "abstained": len(obs)}
    left, right = by_view[views[0]], by_view[views[1]]
    scores, details = [], {}
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            score, detail, reason = _compatibility(a, b, frame.cameras)
            scores.append((score, i, j)); details[f"{i}:{j}"] = {**detail, "reason": reason}
    used_l, used_r, fused = set(), set(), []
    for score, i, j in sorted(scores, reverse=True):
        if not np.isfinite(score) or i in used_l or j in used_r:
            continue
        a, b = left[i], right[j]
        support = np.concatenate([a.support_points, b.support_points], axis=0)
        center = np.average(np.stack([a.center, b.center]), axis=0, weights=[a.confidence, b.confidence])
        covariance = (a.covariance + b.covariance) / 2.0 + np.eye(3) * (np.linalg.norm(a.center - b.center) ** 2 / 4.0)
        metrics = details[f"{i}:{j}"]
        fused.append(WorldDetectionObservation(
            observation_id=f"fused:{a.observation_id}+{b.observation_id}",
            detections=a.detections + b.detections, support_points=support,
            center=center, covariance=covariance,
            confidence=float((a.confidence + b.confidence) / 2.0),
            geometry_reliable=True, fusion_status="FUSED_TWO_VIEW",
            support_points_by_camera={**a.support_points_by_camera, **b.support_points_by_camera},
            reprojection_coverage={k: float(v) for k, v in metrics.items() if k.startswith("coverage_")},
        ))
        used_l.add(i); used_r.add(j)
    # Preserve unmatched single-view observations and make the reason explicit.
    for i, x in enumerate(left):
        if i not in used_l:
            x.fusion_status = "UNPAIRED_SINGLE_VIEW"
            x.rejection_reason = "no_cross_view_geometry_match"
            fused.append(x)
    for j, x in enumerate(right):
        if j not in used_r:
            x.fusion_status = "UNPAIRED_SINGLE_VIEW"
            x.rejection_reason = "no_cross_view_geometry_match"
            fused.append(x)
    return fused, {
        "candidate_pairs": len(scores), "accepted_pairs": len(fused) - (len(left) - len(used_l)) - (len(right) - len(used_r)),
        "rejected_pairs": sum(1 for score, _, _ in scores if not np.isfinite(score)),
        "abstained": (len(left) - len(used_l)) + (len(right) - len(used_r)),
        "candidate_details": details,
    }

