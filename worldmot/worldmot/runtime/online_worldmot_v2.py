"""Causal WorldMOT v2 runtime with joint Persistent World Queries."""

from __future__ import annotations

from copy import copy
import time
from typing import Mapping

import numpy as np

from ..geometry.camera_conditioned_sampling import project_world_query
from ..geometry.world_alignment import AlignmentResult, Sim3, Sim3Aligner
from ..models.geometry_interface import GeometryBackend, GeometryFrame
from ..models.person_detector import PersonDetector
from ..models.world_observation_encoder import MultiViewObservationEncoder
from ..tracking.persistent_query_runtime import PersistentQueryRuntime
from .online_worldmot import OnlineWorldMOT, _transform_camera


class OnlineWorldMOTV2:
    """One Ω forward and one query update per synchronized RGB timestamp."""

    def __init__(self, detector: PersonDetector, geometry: GeometryBackend,
                 query_runtime: PersistentQueryRuntime,
                 observation_encoder: MultiViewObservationEncoder | None = None,
                 aligner: Sim3Aligner | None = None):
        self.detector = detector
        self.geometry = geometry
        self.query_runtime = query_runtime
        self.observation_encoder = observation_encoder or MultiViewObservationEncoder()
        self.aligner = aligner or Sim3Aligner()
        self.scene_id = None
        self.previous: GeometryFrame | None = None
        self.previous_world_points: dict[int, np.ndarray] = {}

    def reset(self, scene_id=None):
        self.scene_id = scene_id
        self.aligner.reset()
        self.previous = None
        self.previous_world_points = {}
        self.query_runtime.reset(scene_id)

    @staticmethod
    def _set_dynamic_masks(frame: GeometryFrame, detections):
        masks = {}
        for view, ds in detections.items():
            shape = np.asarray(frame.images[view]).shape[:2]
            union = np.zeros(shape, dtype=bool)
            for det in ds:
                if det.mask is not None:
                    union |= np.asarray(det.mask, dtype=bool)
                else:
                    x, y, w, h = det.bbox_xywh
                    union[max(0, int(y)):min(shape[0], int(y + h)), max(0, int(x)):min(shape[1], int(x + w))] = True
            masks[int(view)] = union
        frame.dynamic_masks = masks
        frame.detector_outputs = detections

    def _align(self, frame: GeometryFrame) -> AlignmentResult:
        if self.previous is None:
            return self.aligner.initialize()
        common = sorted(set(frame.pointmaps) & set(self.previous.pointmaps))
        if not common:
            return AlignmentResult(False, Sim3(), 0, 0, 0.0, float("inf"), float("inf"), "no_shared_view")
        previous = copy(self.previous)
        previous.pointmaps = self.previous_world_points or self.previous.pointmaps
        src, dst = OnlineWorldMOT._static_correspondences(previous, frame, common[0])
        if len(src) < self.aligner.min_inliers:
            return AlignmentResult(False, Sim3(), len(src), 0, 0.0, float("inf"), float("inf"), "insufficient_static_matches")
        return self.aligner.update(src, dst)

    def step(self, images_by_camera: Mapping[int, np.ndarray], timestamp: int) -> dict:
        started = time.perf_counter()
        detections = self.detector.infer(images_by_camera)
        frame = self.geometry.reconstruct(images_by_camera, int(timestamp))
        self._set_dynamic_masks(frame, detections)
        alignment = self._align(frame)
        aligned = bool(alignment.success and frame.status == "OK")
        transform = alignment.transform if aligned else None
        tokens = self.observation_encoder.encode(
            detections, images_by_camera, frame, int(timestamp), transform,
            detector_feature_maps=getattr(self.detector, "last_feature_maps", {}),
            omega_patch_features=getattr(frame, "omega_patch_features", {}),
        )
        query_result = self.query_runtime.step(tokens, int(timestamp))
        world_cameras = ({v: _transform_camera(c, alignment.transform) for v, c in frame.cameras.items()}
                         if aligned else dict(frame.cameras))
        tracks_2d = {int(v): [] for v in world_cameras}
        if self.query_runtime.state is not None:
            state = self.query_runtime.state
            for slot, gid in enumerate(state.global_ids[0].tolist()):
                if gid < 0:
                    continue
                xyz = state.xyz[0, slot].detach().cpu().numpy()
                for view, camera in world_cameras.items():
                    assigned = query_result.query_to_observations.get(int(slot), [])
                    current_view = any(tokens[o].camera_id == int(view) for o in assigned)
                    projection = project_world_query(
                        camera, xyz, view, gid, current_view,
                        coordinate_system_valid=bool(self.query_runtime.coordinates_calibrated),
                    )
                    tracks_2d[int(view)].append({
                        "global_id": int(gid), "slot_id": int(slot),
                        "uv": projection.uv.tolist() if projection.uv is not None else None,
                        "visible": projection.visible, "evaluation_eligible": projection.evaluation_eligible,
                        "visibility_evidence": projection.reason,
                    })
        if aligned:
            self.previous = frame
            self.previous_world_points = {v: alignment.transform.apply(x.reshape(-1, 3)).reshape(x.shape)
                                          for v, x in frame.pointmaps.items()}
        return {
            "timestamp": int(timestamp), "scene_id": self.scene_id, "causal": True,
            "architecture": "PWQT_JOINT_MULTI_VIEW_PERSISTENT_WORLD_QUERY",
            "detections_by_camera": {str(v): [d.to_dict(False) for d in ds] for v, ds in detections.items()},
            "world_observations": [t.to_dict(False) for t in tokens],
            "query_assignments": {str(o): int(q) for o, q in query_result.assignments.items()},
            "query_to_observations": {str(q): [int(o) for o in obs]
                                       for q, obs in query_result.query_to_observations.items()},
            "global_ids_by_observation": {str(o): int(g) for o, g in query_result.global_ids_by_observation.items()},
            "birth_global_ids": [int(x) for x in query_result.births],
            "tracks_2d_by_camera": {str(v): rows for v, rows in tracks_2d.items()},
            "alignment": {"success": bool(alignment.success), "reason": alignment.reason,
                          "correspondences": int(alignment.correspondences), "inliers": int(alignment.inliers),
                          "inlier_ratio": float(alignment.inlier_ratio), "residual_median": float(alignment.residual_median),
                          "residual_p95": float(alignment.residual_p95), "scale": float(alignment.transform.scale)},
            "geometry_status": frame.status, "omega_feature_status": getattr(frame, "feature_status", "NOT_REQUESTED"),
            "query_runtime": query_result.diagnostics, "query_status": query_result.status,
            "runtime": {"total_s": float(time.perf_counter() - started),
                        "omega_forward_s": float(frame.diagnostics.get("model_only_s", 0.0)),
                        "detector_forward_s": float(sum(self.detector.last_timings.values()))},
        }
