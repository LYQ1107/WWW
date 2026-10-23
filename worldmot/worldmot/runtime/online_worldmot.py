"""Causal online WorldMOT loop."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Mapping, Optional

import numpy as np

from ..evaluation.benchmark_fps import RuntimeTimer, gpu_memory_summary
from ..geometry.reprojection import project_support
from ..geometry.world_alignment import AlignmentResult, Sim3, Sim3Aligner
from ..models.geometry_interface import GeometryBackend, GeometryFrame
from ..models.panoptic_instances import build_world_instances
from ..tracking.world_memory import WorldIdentityMemory


def _transform_camera(camera, transform: Sim3):
    from copy import deepcopy
    out = deepcopy(camera)
    out.world_from_camera = transform.matrix() @ camera.world_from_camera
    return out


class OnlineWorldMOT:
    def __init__(self, geometry: GeometryBackend, min_confidence: float = 1.5,
                 min_points: int = 24, aligner: Optional[Sim3Aligner] = None,
                 memory: Optional[WorldIdentityMemory] = None, person_category_ids=None,
                 device: str = "cuda:0", emit_predictions: bool = False):
        self.geometry = geometry
        self.min_confidence = float(min_confidence)
        self.min_points = int(min_points)
        self.aligner = aligner or Sim3Aligner()
        self.memory = memory or WorldIdentityMemory()
        self.person_category_ids = person_category_ids
        self.device = device
        self.emit_predictions = bool(emit_predictions)
        self.scene_id = None
        self.previous: Optional[GeometryFrame] = None
        self.previous_world_points: dict[int, np.ndarray] = {}
        self.timer = RuntimeTimer()

    def reset(self, scene_id) -> None:
        self.scene_id = scene_id
        self.aligner.reset()
        self.memory.reset()
        self.previous = None
        self.previous_world_points = {}
        self.timer = RuntimeTimer()

    @staticmethod
    def _native_person_mask(frame: GeometryFrame, view: int) -> np.ndarray | None:
        pred = frame.panoptic.get(view, {}) if isinstance(frame.panoptic, Mapping) else {}
        pan = pred.get("pan") if isinstance(pred, dict) else None
        segments = pred.get("segments_info", []) if isinstance(pred, dict) else []
        if pan is None:
            return None
        pan = np.asarray(pan)
        while pan.ndim > 2 and pan.shape[0] == 1:
            pan = pan[0]
        mask = np.zeros(pan.shape[:2], dtype=bool)
        for seg in segments:
            category = str(seg.get("category_name", "")).lower()
            if category and category != "person" and seg.get("category_id") not in (0, None):
                continue
            if "id" in seg:
                mask |= pan == int(seg["id"])
        return mask if mask.any() else None

    @staticmethod
    def _dynamic_at(frame: GeometryFrame, view: int, x: float, y: float) -> bool:
        masks = getattr(frame, "dynamic_masks", {}) or {}
        mask = masks.get(view)
        if mask is None:
            mask = OnlineWorldMOT._native_person_mask(frame, view)
        if mask is None:
            return False
        mask = np.asarray(mask).astype(bool)
        h, w = np.asarray(frame.images[view]).shape[:2]
        if mask.shape != (h, w):
            from PIL import Image
            from ..utils.pillow_compat import RESAMPLING
            mask = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((w, h), RESAMPLING.NEAREST)) > 0
        xi, yi = int(round(x)), int(round(y))
        return bool(0 <= xi < w and 0 <= yi < h and mask[yi, xi])

    @staticmethod
    def _static_correspondences(previous: GeometryFrame, current: GeometryFrame, view: int,
                                max_matches: int = 128) -> tuple[np.ndarray, np.ndarray]:
        """Match static RGB features and sample their two pointmaps.

        This is only a coordinate-frame estimator.  Person masks are excluded
        when the native panoptic result contains them.  No GT is consulted.
        """
        try:
            import cv2
        except Exception:
            return np.empty((0, 3)), np.empty((0, 3))
        a, b = previous.images[view], current.images[view]
        gray_a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create(nfeatures=1200, fastThreshold=12)
        ka, da = orb.detectAndCompute(gray_a, None)
        kb, db = orb.detectAndCompute(gray_b, None)
        if da is None or db is None or not ka or not kb:
            return np.empty((0, 3)), np.empty((0, 3))
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(matcher.match(da, db), key=lambda m: m.distance)[:max_matches]
        pm_a, pm_b = previous.pointmaps[view], current.pointmaps[view]
        cf_a, cf_b = previous.confidences[view], current.confidences[view]
        i2m_a = previous.image_to_model.get(view, np.eye(3))
        i2m_b = current.image_to_model.get(view, np.eye(3))
        h_a, w_a = pm_a.shape[:2]; h_b, w_b = pm_b.shape[:2]
        pa, pb = [], []
        for m in matches:
            # A keypoint is a valid static-background support point only when
            # neither its previous nor current RGB location falls inside a
            # causal person mask.  This filters moving people before Sim(3)
            # RANSAC; no annotation or future frame is consulted.
            if OnlineWorldMOT._dynamic_at(previous, view, *ka[m.queryIdx].pt):
                continue
            if OnlineWorldMOT._dynamic_at(current, view, *kb[m.trainIdx].pt):
                continue
            def sample(kp, pm, cf, mat, h, w, pred):
                p = np.array([kp.pt[0], kp.pt[1], 1.0])
                q = mat @ p; x, y = int(round(q[0] / q[2])), int(round(q[1] / q[2]))
                if x < 0 or y < 0 or x >= w or y >= h:
                    return None
                if not np.isfinite(pm[y, x]).all() or not np.isfinite(cf[y, x]) or cf[y, x] < 1.2:
                    return None
                return pm[y, x]
            xa = sample(ka[m.queryIdx], pm_a, cf_a, i2m_a, h_a, w_a, previous.panoptic.get(view))
            xb = sample(kb[m.trainIdx], pm_b, cf_b, i2m_b, h_b, w_b, current.panoptic.get(view))
            if xa is not None and xb is not None:
                pa.append(xb); pb.append(xa)
        return np.asarray(pa, dtype=np.float64), np.asarray(pb, dtype=np.float64)

    def _align(self, frame: GeometryFrame) -> AlignmentResult:
        if self.previous is None:
            return self.aligner.initialize()
        common = sorted(set(frame.pointmaps) & set(self.previous.pointmaps))
        if not common:
            return AlignmentResult(False, Sim3(), 0, 0, 0.0, float("inf"), float("inf"), "no_shared_view")
        # Current points are mapped to previous world points by the previous
        # frame's accumulated transform before RANSAC.
        # The previous frame's maps are stored in the accumulated world frame;
        # the current maps are still in the current joint reconstruction frame.
        from copy import copy
        previous_world = copy(self.previous)
        previous_world.pointmaps = self.previous_world_points or self.previous.pointmaps
        src, dst = self._static_correspondences(previous_world, frame, common[0])
        if len(src) < self.aligner.min_inliers:
            return AlignmentResult(False, Sim3(), len(src), 0, 0.0, float("inf"), float("inf"), "insufficient_static_matches")
        return self.aligner.update(src, dst)

    def step(self, images_by_camera: Mapping[int, np.ndarray], timestamp: int) -> dict:
        with self.timer.stage("total"):
            with self.timer.stage("geometry"):
                frame = self.geometry.reconstruct(images_by_camera, int(timestamp))
            with self.timer.stage("world_alignment"):
                alignment = self._align(frame)
            aligned = bool(alignment.success and frame.status not in ("INVALID_WORLD_GEOMETRY",))
            instances = []
            if aligned:
                with self.timer.stage("instance_build"):
                    instances = build_world_instances(frame.pointmaps, frame.confidences, frame.panoptic,
                                                      frame.query_features, self.min_confidence, self.min_points,
                                                      person_category_ids=self.person_category_ids)
                    instances = [x.transformed(alignment.transform) for x in instances]
            with self.timer.stage("identity_memory"):
                tracks = self.memory.step(instances, int(timestamp), geometry_aligned=aligned)
            world_cameras = {v: _transform_camera(c, alignment.transform) for v, c in frame.cameras.items()} if aligned else dict(frame.cameras)
            tracks_2d = {}
            with self.timer.stage("reprojection"):
                suppressed_predictions = []
                for view, camera in world_cameras.items():
                    rows = []
                    for tr in tracks:
                        if tr.status.value == "DELETED" or len(tr.support_points) == 0:
                            continue
                        if not tr.observed_this_step or tr.status.value == "LOST":
                            suppressed_predictions.append({
                                "global_id": int(tr.global_id), "view": int(view),
                                "reason": "LOST_OR_PURE_PREDICTION_SUPPRESSED",
                                "missed": int(tr.missed), "status": tr.status.value,
                            })
                            if not self.emit_predictions:
                                continue
                        p = project_support(tr.support_points, camera)
                        native_observed = bool(view in tr.masks_by_camera) and tr.observed_this_step
                        # ``visible`` is evaluation-eligible only when this
                        # synchronized frame has an actual target-view mask.
                        # A cross-view projection remains available as a
                        # diagnostic but is never presented as a real target
                        # observation.
                        rows.append({
                            "global_id": int(tr.global_id), "visible": bool(p.visible and native_observed),
                            "projected_visible": bool(p.visible),
                            "bbox_xywh": list(p.bbox_xywh) if p.bbox_xywh is not None else None,
                            "visible_fraction": float(p.visible_fraction),
                            "native_view_mask": native_observed,
                            "visibility_state": "CURRENT_MASK_AND_PROJECTION" if native_observed else "CURRENT_OTHER_VIEW_ONLY",
                            "visibility_evidence": "current_target_mask" if native_observed else "projection_only",
                            "projected_from_world": True,
                            "projection_refined_by_target_view": False,
                            "evaluation_eligible": native_observed,
                            "status": tr.status.value,
                        })
                    tracks_2d[int(view)] = rows
            if aligned:
                self.previous = frame
                self.previous_world_points = {v: alignment.transform.apply(x.reshape(-1, 3)).reshape(x.shape)
                                              for v, x in frame.pointmaps.items()}
            self.timer.add("preprocess", float(frame.diagnostics.get("preprocess_s", 0.0)))
            self.timer.add("model_only", float(frame.diagnostics.get("model_only_s", 0.0)))
            self.timer.add("panoptic", float(frame.diagnostics.get("panoptic_s", 0.0)))
            self.timer.add("pointmap_postprocess", float(frame.diagnostics.get("pointmap_postprocess_s", 0.0)))
        return {
            "timestamp": int(timestamp), "scene_id": self.scene_id,
            "tracks_3d": [x.to_dict() for x in tracks], "tracks_2d_by_camera": tracks_2d,
            "suppressed_predictions": suppressed_predictions,
            "geometry_status": frame.status, "alignment": {
                "success": bool(alignment.success), "reason": alignment.reason,
                "correspondences": int(alignment.correspondences), "inliers": int(alignment.inliers),
                "inlier_ratio": float(alignment.inlier_ratio), "residual_median": float(alignment.residual_median),
                "residual_p95": float(alignment.residual_p95), "scale": float(alignment.transform.scale),
            },
            "runtime": {"stages": self.timer.summary(),
                        "frame_stages": dict(self.timer.last_stages),
                        "frame_total_s": float(self.timer.last_stages.get("total", 0.0)),
                        "gpu": gpu_memory_summary(self.device)},
            "proposal_count": len(instances), "causal": True,
        }
