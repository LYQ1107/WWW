"""Build causal multi-camera observation tokens for PWQT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..geometry.relative_geometry_encoding import encode_relative_geometry, fit_geometry_normalization
from .detection_to_world import lift_detection
from .omega_object_features import FrozenObjectFeatureExtractor
from .person_detector import PersonDetection
from ..geometry.reprojection import project_support


@dataclass
class WorldObservationToken:
    camera_id: int
    timestamp: int
    detection_id: int
    image_feature: np.ndarray
    omega_feature: np.ndarray
    xyz_world: np.ndarray
    covariance: np.ndarray
    bbox: np.ndarray
    mask_feature: np.ndarray
    detection_score: float
    geometry_reliability: float
    geometry_feature: np.ndarray
    valid: bool
    source: str = ""
    diagnostics: dict = field(default_factory=dict)

    @property
    def feature(self) -> np.ndarray:
        if len(self.image_feature) == 512:
            return np.concatenate([self.image_feature, self.geometry_feature,
                                   np.asarray([self.detection_score, self.geometry_reliability], np.float32)])
        return np.concatenate([
            np.asarray(self.image_feature, dtype=np.float32).reshape(-1),
            np.asarray(self.omega_feature, dtype=np.float32).reshape(-1),
            np.asarray(self.geometry_feature, dtype=np.float32).reshape(-1),
            np.asarray([self.detection_score, self.geometry_reliability], dtype=np.float32),
        ])

    def to_dict(self, include_features: bool = False) -> dict:
        out = {
            "camera_id": int(self.camera_id), "timestamp": int(self.timestamp),
            "detection_id": int(self.detection_id), "xyz_world": np.asarray(self.xyz_world, dtype=float).tolist(),
            "covariance": np.asarray(self.covariance, dtype=float).tolist(),
            "bbox": np.asarray(self.bbox, dtype=float).tolist(),
            "detection_score": float(self.detection_score),
            "geometry_reliability": float(self.geometry_reliability), "valid": bool(self.valid),
            "source": self.source, "diagnostics": dict(self.diagnostics),
        }
        if include_features:
            out.update({"image_feature": np.asarray(self.image_feature, dtype=float).tolist(),
                        "omega_feature": np.asarray(self.omega_feature, dtype=float).tolist(),
                        "mask_feature": np.asarray(self.mask_feature, dtype=float).tolist(),
                        "geometry_feature": np.asarray(self.geometry_feature, dtype=float).tolist()})
        return out


def _mask_descriptor(mask: np.ndarray | None, bbox: Sequence[float], shape: tuple[int, int], size: int = 8) -> np.ndarray:
    from PIL import Image
    from ..utils.pillow_compat import RESAMPLING

    h, w = shape
    if mask is not None and np.asarray(mask).shape == (h, w):
        m = np.asarray(mask, dtype=np.uint8)
        x, y, bw, bh = [float(v) for v in bbox]
        x0, y0 = max(0, int(x)), max(0, int(y)); x1, y1 = min(w, int(x + bw)), min(h, int(y + bh))
        m = m[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else np.zeros((1, 1), dtype=np.uint8)
    else:
        m = np.ones((max(1, int(bbox[3])), max(1, int(bbox[2]))), dtype=np.uint8)
    return (np.asarray(Image.fromarray(m * 255).resize((size, size), RESAMPLING.NEAREST), dtype=np.float32) / 255.0).reshape(-1)


def _apply_sim3(transform, xyz: np.ndarray) -> np.ndarray:
    if transform is None:
        return np.asarray(xyz, dtype=np.float32)
    return np.asarray(transform.apply(np.asarray(xyz, dtype=np.float64).reshape(1, 3))[0], dtype=np.float32)


class MultiViewObservationEncoder:
    """Encode detections independently; no A/B pair is pre-fused."""

    def __init__(self, feature_extractor: FrozenObjectFeatureExtractor | None = None,
                 min_confidence: float = 1.5, min_points: int = 16):
        self.feature_extractor = feature_extractor or FrozenObjectFeatureExtractor()
        self.min_confidence = float(min_confidence)
        self.min_points = int(min_points)
        self.normalization = None

    def reset(self):
        self.normalization = None

    def encode(
        self,
        detections: Mapping[int, Sequence[PersonDetection]],
        images: Mapping[int, np.ndarray],
        geometry_frame,
        timestamp: int,
        world_transform=None,
        detector_feature_maps: Mapping[int, np.ndarray] | None = None,
        omega_patch_features: Mapping[int, np.ndarray] | None = None,
    ) -> list[WorldObservationToken]:
        source_features = self.feature_extractor.extract(
            detections, images, detector_feature_maps, omega_patch_features,
        )
        raw_centers = []
        lifted = []
        for view, ds in sorted(detections.items()):
            for det in ds:
                obs = lift_detection(det, geometry_frame, self.min_confidence, self.min_points)
                lifted.append((int(view), det, obs))
                if np.isfinite(obs.center).all():
                    raw_centers.append(_apply_sim3(world_transform, obs.center))
        if self.normalization is None and raw_centers:
            self.normalization = fit_geometry_normalization(np.asarray(raw_centers, dtype=np.float32))
        normalization = self.normalization or fit_geometry_normalization(np.empty((0, 3)))
        out: list[WorldObservationToken] = []
        for view, det, obs in lifted:
            feature = source_features[(int(view), int(det.detection_id))]
            xyz = _apply_sim3(world_transform, obs.center) if np.isfinite(obs.center).all() else np.full(3, np.nan, np.float32)
            camera = getattr(geometry_frame, "cameras", {}).get(int(view))
            camera_pose = getattr(camera, "world_from_camera", None)
            if camera_pose is not None and world_transform is not None:
                camera_pose = np.array(camera_pose, copy=True)
                camera_pose[:3, :3] = world_transform.rotation @ camera_pose[:3, :3]
                camera_pose[:3, 3] = world_transform.apply(camera_pose[:3, 3][None])[0]
            world_cov = np.asarray(obs.covariance, dtype=np.float32)
            if world_transform is not None:
                rot = world_transform.rotation
                world_cov = world_transform.scale ** 2 * rot @ world_cov @ rot.T
            geom = encode_relative_geometry(
                xyz if np.isfinite(xyz).all() else np.zeros(3, dtype=np.float32),
                covariance=world_cov, camera_pose=camera_pose,
                geometry_reliability=1.0 if obs.geometry_reliable else 0.0,
                visible=1.0, normalization=normalization,
            )
            if world_transform is None:
                geom[:] = 0
            projection = project_support(obs.support_points, camera) if camera is not None else None
            out.append(WorldObservationToken(
                camera_id=int(view), timestamp=int(timestamp), detection_id=int(det.detection_id),
                image_feature=feature.appearance_feature, omega_feature=feature.omega_feature,
                xyz_world=xyz, covariance=world_cov,
                bbox=np.asarray(det.bbox_xywh, dtype=np.float32),
                mask_feature=_mask_descriptor(det.mask, det.bbox_xywh, np.asarray(images[view]).shape[:2]),
                detection_score=float(det.score), geometry_reliability=float(obs.geometry_reliable),
                geometry_feature=geom, valid=bool(feature.valid),
                source=feature.source,
                diagnostics={"lift_rejection_reason": obs.rejection_reason,
                             "fusion_preassignment_forbidden": True,
                             "omega_feature_available": bool(len(feature.omega_feature)),
                             "rtmdet_feature_available": bool(len(feature.detector_feature)),
                             "normalization_origin": normalization.origin.tolist(),
                             "normalization_scale": float(normalization.scale),
                             "geometry_coordinate_valid": bool(world_transform is not None and obs.geometry_reliable)},
            ))
            out[-1].diagnostics["pure_3d_bbox_xywh"] = list(projection.bbox_xywh) if projection and projection.bbox_xywh else None
        return out


def observations_to_tensors(tokens: Sequence[WorldObservationToken],
                            omega_dim: int = 64, device=None, dtype=None):
    """Pack tokens into the fixed PWQT input contract.

    Omega patch channels depend on the selected intermediate layer.  They are
    deterministically truncated/padded to ``omega_dim`` at this boundary;
    the uncompressed source feature remains in the audit record.
    """
    import torch

    device = device or "cpu"
    dtype = dtype or torch.float32
    obs = []
    geo = []
    camera_ids = []
    for token in tokens:
        if len(token.image_feature) == 512:
            obs.append(token.feature)
            geo.append(token.geometry_feature)
            camera_ids.append(int(token.camera_id))
            continue
        omega = np.asarray(token.omega_feature, dtype=np.float32).reshape(-1)
        if len(omega) < omega_dim:
            omega = np.pad(omega, (0, omega_dim - len(omega)))
        else:
            omega = omega[:omega_dim]
        obs.append(np.concatenate([np.asarray(token.image_feature, dtype=np.float32).reshape(-1), omega,
                                   np.asarray(token.geometry_feature, dtype=np.float32).reshape(-1),
                                   np.asarray([token.detection_score, token.geometry_reliability], dtype=np.float32)]))
        geo.append(np.asarray(token.geometry_feature, dtype=np.float32).reshape(-1))
        camera_ids.append(int(token.camera_id))
    if not obs:
        return (torch.empty((1, 0, 256 + omega_dim + 18 + 2), device=device, dtype=dtype),
                torch.empty((1, 0, 18), device=device, dtype=dtype), [], torch.empty((1, 0), dtype=torch.bool, device=device))
    return (torch.as_tensor(np.stack(obs), device=device, dtype=dtype)[None],
            torch.as_tensor(np.stack(geo), device=device, dtype=dtype)[None], camera_ids,
            torch.as_tensor([bool(x.valid) for x in tokens], device=device, dtype=torch.bool)[None])
