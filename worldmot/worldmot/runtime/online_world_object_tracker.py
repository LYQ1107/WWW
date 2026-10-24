"""Online tracker whose persistent unit is a shared world object."""

from __future__ import annotations

from typing import Mapping
import time

import numpy as np

from ..models.detection_to_world import lift_detection
from ..models.multi_view_world_object import JointWorldObjectBuilder
from ..models.world_object_support import extract_world_object_support
from ..tracking.persistent_world_object_bank import PersistentWorldObjectBank
from ..geometry.dual_view_static_alignment import DualCameraStaticAligner
from ..runtime.detection_first import DetectionFirstWorldMOT
from ..geometry.reprojection import project_support


class OnlineWorldObjectTracker:
    """Frozen GMT + Omega front-end with an explicit world-object bank."""

    def __init__(self, detector, geometry, device="cuda:0", min_confidence=1.5,
                 min_points=8, use_anchor=False):
        self.detector = detector; self.geometry = geometry; self.device = str(device)
        self.min_confidence = float(min_confidence); self.min_points = int(min_points)
        self.aligner = DualCameraStaticAligner(min_inliers=6)
        self.builder = JointWorldObjectBuilder()
        self.bank = PersistentWorldObjectBank()
        self.use_anchor = bool(use_anchor)
        self.scene_id = None; self.previous = None; self.previous_world = {}
        self.last_timings = {}

    def reset(self, scene_id):
        self.scene_id = scene_id; self.aligner.reset(); self.bank.reset()
        self.previous = None; self.previous_world = {}; self.last_timings = {}

    def _align(self, frame, detections):
        DetectionFirstWorldMOT._set_dynamic_masks(frame, detections)
        if self.previous is None:
            result = self.aligner.initialize(); return result
        old = self.previous
        old.world_pointmaps = self.previous_world
        return self.aligner.update(old, frame, sorted(frame.pointmaps))

    def step(self, images_by_camera: Mapping[int, np.ndarray], timestamp: int) -> dict:
        start = time.perf_counter(); self.last_timings = {}
        t = time.perf_counter(); detections = self.detector.infer(images_by_camera)
        self.last_timings["detector"] = time.perf_counter() - t
        t = time.perf_counter(); frame = self.geometry.reconstruct(images_by_camera, int(timestamp))
        self.last_timings["omega"] = time.perf_counter() - t
        alignment = self._align(frame, detections)
        aligned = bool(alignment.result.success and frame.status == "OK")
        observations = []
        t = time.perf_counter()
        for view, ds in sorted(detections.items()):
            pm = frame.pointmaps[int(view)]; cf = frame.confidences[int(view)]
            transform = alignment.result.transform if aligned else self.aligner.transform
            for det in ds:
                sup = extract_world_object_support(pm, cf, det.bbox_xywh,
                                                   min_confidence=self.min_confidence)
                if aligned:
                    box = transform.apply(sup.box_points) if len(sup.box_points) else sup.box_points
                    fg = transform.apply(sup.foreground_points) if len(sup.foreground_points) else sup.foreground_points
                    center = transform.apply(sup.center[None])[0] if np.isfinite(sup.center).all() else sup.center
                    from ..models.world_object_support import WorldObjectSupport
                    sup = WorldObjectSupport(box, fg, center,
                                             transform.scale ** 2 * transform.rotation @ sup.covariance @ transform.rotation.T,
                                             sup.quality, sup.foreground_fraction, sup.ambiguity,
                                             sup.confidence_median, sup.visible_fraction, sup.diagnostics)
                class O: pass
                o = O(); o.camera_id = int(view); o.detection_id = int(det.detection_id); o.support = sup
                o.feature = getattr(det, "detector_roi_feature", np.asarray([det.score], np.float32))
                o.bbox_xywh = det.bbox_xywh; o.score = det.score
                observations.append(o)
        self.last_timings["support"] = time.perf_counter() - t
        hypotheses, formation = self.builder.build(observations)
        t = time.perf_counter(); objects, bank_diag = self.bank.step(hypotheses, int(timestamp)); self.last_timings["bank"] = time.perf_counter() - t
        tracks = {int(x.global_id): x for x in objects}
        tracks_2d = {int(v): [] for v in images_by_camera}
        for j, hyp in enumerate(hypotheses):
            gid = self.bank.last_assignments.get(j)
            if gid is None: continue
            for obs in hyp.observations:
                tracks_2d[int(obs.camera_id)].append({"global_id": int(gid), "bbox_xywh": list(obs.bbox_xywh),
                                                      "visible": True, "evaluation_eligible": True,
                                                      "output_source": "CURRENT_GMT_DETECTION_WORLD_OBJECT"})
        if aligned:
            self.previous = frame
            self.previous_world = {v: alignment.result.transform.apply(x.reshape(-1, 3)).reshape(x.shape)
                                   for v, x in frame.pointmaps.items()}
        self.last_timings["total"] = time.perf_counter() - start
        return {"timestamp": int(timestamp), "scene_id": self.scene_id,
                "detections_by_camera": {str(v): [d.to_dict(False) for d in ds] for v, ds in detections.items()},
                "tracks_3d": [{"global_id": int(x.global_id), "center": x.center.tolist(),
                               "status": x.status, "visible_cameras": list(x.visible_cameras)} for x in objects],
                "tracks_2d_by_camera": tracks_2d, "world_objects": [
                    {"object_id": int(h.object_id), "global_id": self.bank.last_assignments.get(i),
                     "cameras": list(h.cameras), "center": h.center.tolist(), "status": h.status,
                     "score": float(h.score), "diagnostics": h.diagnostics}
                    for i, h in enumerate(hypotheses)],
                "alignment": {"success": aligned, "reason": alignment.result.reason,
                              "inliers": int(alignment.result.inliers), "inlier_ratio": float(alignment.result.inlier_ratio),
                              "residual_p95": float(alignment.result.residual_p95),
                              "per_camera": {str(v): {"success": bool(r.success), "inliers": int(r.inliers),
                                                       "residual_p95": float(r.residual_p95)}
                                              for v, r in alignment.per_camera.items()},
                              "roundtrip_error": float(alignment.roundtrip_error)},
                "formation": formation, "bank": bank_diag, "runtime": {"stages": dict(self.last_timings)},
                "causal": True, "output_policy": "CURRENT_DETECTION_ONLY"}
