"""Camera parameters and explicit image/model coordinate transforms."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _as_homogeneous(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.shape == (4, 4):
        return x
    raise ValueError(f"expected 4x4 transform, got {x.shape}")


@dataclass
class CameraModel:
    """Pinhole camera with a camera-to-world pose.

    ``image_to_model`` maps original image pixels to the network's model
    pixels.  The intrinsic matrix is expressed in model pixels.  Projection
    returns original image coordinates, which keeps crop/resize handling
    visible and testable.
    """

    intrinsic_model: np.ndarray
    world_from_camera: np.ndarray
    original_size: tuple[int, int]  # (height, width)
    model_size: tuple[int, int]     # (height, width)
    image_to_model: np.ndarray | None = None
    name: str = ""

    def __post_init__(self) -> None:
        self.intrinsic_model = np.asarray(self.intrinsic_model, dtype=np.float64).reshape(3, 3)
        self.world_from_camera = _as_homogeneous(self.world_from_camera)
        if self.image_to_model is None:
            h, w = self.original_size
            hm, wm = self.model_size
            self.image_to_model = np.array([[wm / w, 0.0, 0.0], [0.0, hm / h, 0.0], [0, 0, 1]], dtype=np.float64)
        self.image_to_model = np.asarray(self.image_to_model, dtype=np.float64).reshape(3, 3)
        if not np.isfinite(self.intrinsic_model).all() or not np.isfinite(self.world_from_camera).all():
            raise ValueError("camera parameters contain non-finite values")

    @property
    def camera_from_world(self) -> np.ndarray:
        return np.linalg.inv(self.world_from_camera)

    @property
    def intrinsic_original(self) -> np.ndarray:
        return np.linalg.inv(self.image_to_model) @ self.intrinsic_model

    def project(self, points_world: np.ndarray, clip: bool = False) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        ph = np.concatenate([points, np.ones((len(points), 1))], axis=1)
        cam = (self.camera_from_world @ ph.T).T[:, :3]
        z = cam[:, 2]
        valid = np.isfinite(cam).all(axis=1) & (z > 1e-8)
        uv_model = np.full((len(points), 2), np.nan, dtype=np.float64)
        if valid.any():
            q = (self.intrinsic_model @ cam[valid].T).T
            uv_model[valid] = q[:, :2] / q[:, 2:3]
        try:
            model_to_image = np.linalg.inv(self.image_to_model)
        except np.linalg.LinAlgError:
            valid[:] = False
            return uv_model, valid
        uv_h = np.concatenate([uv_model, np.ones((len(points), 1))], axis=1)
        uv_img_h = (model_to_image @ uv_h.T).T
        uv = uv_img_h[:, :2] / np.maximum(uv_img_h[:, 2:3], 1e-12)
        if clip:
            h, w = self.original_size
            valid &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        return uv, valid

    def to_dict(self) -> dict:
        return {
            "intrinsic_model": self.intrinsic_model.tolist(),
            "world_from_camera": self.world_from_camera.tolist(),
            "original_size": list(self.original_size),
            "model_size": list(self.model_size),
            "image_to_model": self.image_to_model.tolist(),
            "name": self.name,
        }
