"""Detection-first RTMDet + VGGT-Ω WorldMOT runtime.

This is intentionally separate from ``OnlineWorldMOT``'s PanSt3R path so the
ablation cannot accidentally reuse native joint instances as Omega detections.
"""

from __future__ import annotations

from copy import copy
from typing import Mapping, Optional
import time

import numpy as np

from ..evaluation.benchmark_fps import RuntimeTimer, gpu_memory_summary
from ..geometry.reprojection import project_support
from ..geometry.world_alignment import AlignmentResult, Sim3, Sim3Aligner
from ..models.detection_to_world import lift_detection, fuse_two_views
from ..models.geometry_interface import GeometryBackend, GeometryFrame
from ..models.person_detector import PersonDetector
from ..models.panoptic_instances import WorldInstance
from ..tracking.world_memory import WorldIdentityMemory
from .online_worldmot import OnlineWorldMOT, _transform_camera


class DetectionFirstWorldMOT:
    @staticmethod
    def _current_detection_support(observations, instances, assignments):
        support = {}
        for index, global_id in assignments.items():
            observation = observations[int(instances[index].proposal_id)]
            for detection in observation.detections:
                support[int(global_id), int(detection.view_id)] = {
                    "has_real_mask": detection.mask is not None and bool(np.asarray(detection.mask, bool).any()),
                }
        return support

    @staticmethod
    def _visibility_evidence(global_id, view, current_support):
        current = current_support.get((int(global_id), int(view)))
        native = current is not None
        mask = bool(current and current["has_real_mask"])
        return {
            "native_view_detection": native,
            "native_view_mask": mask,
            "evaluation_eligible": native,
            "visibility_state": ("CURRENT_MASK_AND_PROJECTION" if mask else
                                 "CURRENT_BBOX_AND_PROJECTION" if native else "CURRENT_OTHER_VIEW_ONLY"),
            "visibility_evidence": ("current_target_mask" if mask else
                                    "current_target_detection_bbox" if native else "projection_only"),
        }

    def __init__(self, detector: PersonDetector, geometry: GeometryBackend,
                 min_confidence: float = 1.5, min_points: int = 16,
                 aligner: Optional[Sim3Aligner] = None,
                 memory: Optional[WorldIdentityMemory] = None,
                 device: str = "cuda:0"):
        self.detector = detector
        self.detector_frontend = detector.manifest().get("source", "rtmdet")
        self.geometry = geometry
        self.min_confidence = float(min_confidence)
        self.min_points = int(min_points)
        self.aligner = aligner or Sim3Aligner()
        self.memory = memory or WorldIdentityMemory()
        self.device = device
        self.scene_id = None
        self.previous: GeometryFrame | None = None
        self.previous_world_points: dict[int, np.ndarray] = {}
        self.timer = RuntimeTimer()

    def reset(self, scene_id) -> None:
        self.scene_id = scene_id
        self.aligner.reset(); self.memory.reset()
        self.previous = None; self.previous_world_points = {}
        self.timer = RuntimeTimer()

    def _align(self, frame: GeometryFrame) -> AlignmentResult:
        if self.previous is None:
            return self.aligner.initialize()
        common = sorted(set(frame.pointmaps) & set(self.previous.pointmaps))
        if not common:
            return AlignmentResult(False, Sim3(), 0, 0, 0.0, float("inf"), float("inf"), "no_shared_view")
        previous_world = copy(self.previous)
        previous_world.pointmaps = self.previous_world_points or self.previous.pointmaps
        src, dst = OnlineWorldMOT._static_correspondences(previous_world, frame, common[0])
        if len(src) < self.aligner.min_inliers:
            return AlignmentResult(False, Sim3(), len(src), 0, 0.0, float("inf"), float("inf"),
                                   "insufficient_static_matches")
        return self.aligner.update(src, dst)

    @staticmethod
    def _set_dynamic_masks(frame: GeometryFrame, detections: Mapping[int, list]) -> None:
        masks = {}
        for view, dets in detections.items():
            shape = np.asarray(frame.images[view]).shape[:2]
            union = np.zeros(shape, dtype=bool)
            for det in dets:
                if det.mask is not None:
                    union |= np.asarray(det.mask).astype(bool)
                else:
                    x, y, w, h = det.bbox_xywh
                    x0, y0 = max(0, int(x)), max(0, int(y)); x1, y1 = min(shape[1], int(x+w)), min(shape[0], int(y+h))
                    union[y0:y1, x0:x1] = True
            masks[int(view)] = union
        frame.dynamic_masks = masks
        frame.detector_outputs = detections

    def step(self, images_by_camera: Mapping[int, np.ndarray], timestamp: int) -> dict:
        t0 = time.perf_counter()
        self.timer.last_stages = {}
        with self.timer.stage("total"):
            with self.timer.stage("person_detection"):
                detections = self.detector.infer(images_by_camera)
            frame = self.geometry.reconstruct(images_by_camera, int(timestamp))
            self._set_dynamic_masks(frame, detections)
            with self.timer.stage("world_alignment"):
                alignment = self._align(frame)
            aligned = bool(alignment.success and frame.status == "OK")
            observations = []
            fusion_diag = {}
            if aligned:
                with self.timer.stage("detection_to_world"):
                    lifted = []
                    for view in sorted(detections):
                        for det in detections[view]:
                            lifted.append(lift_detection(det, frame, self.min_confidence, self.min_points))
                    observations, fusion_diag = fuse_two_views(lifted, frame)
                with self.timer.stage("instance_build"):
                    instances = [x.to_world_instance(i) for i, x in enumerate(observations)
                                 if x.geometry_reliable]
                    instances = [x.transformed(alignment.transform) for x in instances]
            else:
                instances = []
                fusion_diag = {"reason": alignment.reason}
            with self.timer.stage("identity_memory"):
                tracks = self.memory.step(instances, int(timestamp), geometry_aligned=aligned)
            current_support = self._current_detection_support(
                observations, instances, self.memory.last_instance_to_global_id)
            world_cameras = ({v: _transform_camera(c, alignment.transform) for v, c in frame.cameras.items()}
                             if aligned else dict(frame.cameras))
            tracks_2d = {}
            with self.timer.stage("reprojection"):
                for view, camera in world_cameras.items():
                    rows = []
                    for tr in tracks:
                        if tr.status.value == "DELETED" or not tr.observed_this_step or len(tr.support_points) == 0:
                            continue
                        support = getattr(tr, "support_points_by_camera", {}).get(
                            int(view), tr.support_points)
                        p = project_support(support, camera)
                        evidence = self._visibility_evidence(tr.global_id, view, current_support)
                        rows.append({
                            "global_id": int(tr.global_id),
                            "visible": bool(p.visible and evidence["evaluation_eligible"]),
                            "projected_visible": bool(p.visible),
                            "bbox_xywh": list(p.bbox_xywh) if p.bbox_xywh is not None else None,
                            "visible_fraction": float(p.visible_fraction),
                            **evidence,
                            "projected_from_world": True,
                            "status": tr.status.value,
                        })
                    tracks_2d[int(view)] = rows
            if aligned:
                self.previous = frame
                self.previous_world_points = {
                    v: alignment.transform.apply(x.reshape(-1, 3)).reshape(x.shape)
                    for v, x in frame.pointmaps.items()
                }
            self.timer.add("preprocess", float(frame.diagnostics.get("preprocess_s", 0.0)))
            self.timer.add("omega_forward", float(frame.diagnostics.get("model_only_s", 0.0)))
            self.timer.add("detector_forward", float(sum(self.detector.last_timings.values())))
        obs_rows = []
        # Export actual memory decisions, preserving the original v1 output.
        # The observed-box control isolates projection-box effects from IDs.
        proposal_to_gid = {int(instances[j].proposal_id): int(gid)
                           for j, gid in self.memory.last_instance_to_global_id.items()}
        observation_boxes = {str(v): [] for v in images_by_camera}
        for proposal_id, obs in enumerate(observations):
            row = obs.to_dict(include_support=True)
            gid = proposal_to_gid.get(proposal_id)
            if gid is not None:
                row["global_id"] = gid
                row["identity_assignment_source"] = "ACTUAL_MEMORY_HUNGARIAN_OR_BIRTH"
                for det in obs.detections:
                    observation_boxes[str(det.view_id)].append({"global_id":gid,
                        "bbox_xywh":list(det.bbox_xywh),"score":float(det.score),
                        "visible":True,"evaluation_eligible":True,
                        "output_source":"CURRENT_DETECTOR_BBOX_ACTUAL_V1_ID"})
            obs_rows.append(row)
        return {
            "timestamp": int(timestamp), "scene_id": self.scene_id,
            "detections_by_camera": {str(v): [x.to_dict(include_mask=False) for x in ds] for v, ds in detections.items()},
            "world_observations": obs_rows,
            "tracks_3d": [x.to_dict() for x in tracks],
            "tracks_2d_by_camera": tracks_2d,
            "tracks_2d_observation_boxes_by_camera": observation_boxes,
            "geometry_status": frame.status, "alignment": {
                "success": bool(alignment.success), "reason": alignment.reason,
                "correspondences": int(alignment.correspondences), "inliers": int(alignment.inliers),
                "inlier_ratio": float(alignment.inlier_ratio), "residual_median": float(alignment.residual_median),
                "residual_p95": float(alignment.residual_p95), "scale": float(alignment.transform.scale),
            },
            "fusion": fusion_diag,
            "runtime": {"stages": self.timer.summary(),
                        "frame_stages": dict(self.timer.last_stages),
                        "frame_total_s": float(self.timer.last_stages.get("total", 0.0)),
                        "gpu": gpu_memory_summary(self.device)},
            "proposal_count": len(instances), "causal": True,
            "ablation": ("A_DETECT_FIRST_GMT_PRE_GTA_OMEGA512" if self.detector_frontend == "gmt_pre_gta_detector"
                         else "A_DETECT_FIRST_RTMDET_INS_TINY_OMEGA512"),
            "detector_status": self.detector.manifest(),
            "omega_status": self.geometry.diagnostics() if hasattr(self.geometry, "diagnostics") else {},
        }

