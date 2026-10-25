"""Live/replay adapters feeding exactly the same GT-free tracking core."""
from __future__ import annotations
import numpy as np
from ..models.world_object_observation import WorldObjectObservation, WORLD_ALIGNMENT_FAILED
from ..models.world_object_support import WorldObjectSupport, extract_world_object_support, map_bbox_to_model

TARGET_KEYS = {"identity", "gt_id", "gt_identity", "instance_id", "track_id", "global_id", "annotations", "targets"}


def replay_observations(frame):
    observations = []
    for row in frame["rows"]:
        if TARGET_KEYS.intersection(row):
            raise ValueError("GT_IDENTITY_FORBIDDEN_IN_INFERENCE_OBSERVATION")
        valid = bool(row.get("geometry_valid", False))
        support = WorldObjectSupport(np.asarray(row["box"]), np.asarray(row["foreground"]),
            np.asarray(row["center"]), np.asarray(row["cov"]), float(row["quality"]),
            float(row["foreground_fraction"]), str(row.get("geometry_status", "UNVERIFIED")),
            coordinate_frame="PERSISTENT_WORLD", depth_source="CACHE_V2_CAMERA_DEPTH" if valid else "UNVERIFIED")
        observations.append(WorldObjectObservation(row["view"], row["detection_id"], row["bbox"], row["score"],
            row.get("roi_feature"), row.get("feature_valid", False), support.center, support.covariance,
            support, valid, support.quality if valid else 0.0, frame["timestamp"],
            geometry_status=row.get("geometry_status", "VISUAL_ONLY_FALLBACK"), source=frame.get("source", {})))
    return sorted(observations, key=lambda o: o.key)


def live_observations(detections, frame, world_transform, *, min_confidence=1.5, use_geometry=True):
    if int(world_transform.timestamp)!=int(frame.timestamp):
        raise ValueError("STALE_WORLD_FRAME_TRANSFORM")
    observations = []
    for camera, ds in sorted(detections.items()):
        for det in ds:
            if int(det.view_id)!=int(camera):
                raise ValueError("DETECTION_CAMERA_SOURCE_MISMATCH")
            support = None
            valid = False
            status = "VISUAL_ONLY_FALLBACK"
            if use_geometry and world_transform.valid and frame.status == "OK" and camera in frame.cameras:
                pointmap, conf = frame.pointmaps[camera], frame.confidences[camera]
                if camera not in frame.image_to_model:
                    raise ValueError("IMAGE_TO_MODEL_MAPPING_REQUIRED")
                bbox = map_bbox_to_model(det.bbox_xywh, frame.image_to_model[camera], pointmap.shape[:2])
                support = extract_world_object_support(pointmap, conf, bbox,
                    min_confidence=min_confidence, camera_from_world=frame.cameras[camera].camera_from_world)
                support = world_transform.support(support)
                valid = support.reliable
                status = ("GEOMETRY_VALID" if valid and support.quality >= 0.6 else
                          "GEOMETRY_LOW_CONFIDENCE" if valid else "SPATIAL_SUPPORT_AMBIGUOUS")
            elif use_geometry:
                status = WORLD_ALIGNMENT_FAILED
            feature = getattr(det, "detector_roi_feature", None)
            observations.append(WorldObjectObservation(int(camera), det.detection_id, det.bbox_xywh, det.score,
                feature, feature is not None,
                support.center if support else np.full(3, np.nan),
                support.covariance if support else np.eye(3)*1e6, support, valid,
                support.quality if valid else 0.0, int(frame.timestamp), geometry_status=status))
    return sorted(observations, key=lambda o: o.key)
