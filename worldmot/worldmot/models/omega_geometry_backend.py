"""Official VGGT-Ω 512 geometry backend for the WorldMOT ablation.

This adapter accepts two complete RGB images and exposes the released
camera/depth outputs as WorldMOT pointmaps.  It does not import GMT or any
identity model.  The checkpoint and source hashes are checked before the
model is used so a similarly named or random checkpoint cannot enter the
experiment silently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import sys
import time
from typing import Mapping

import numpy as np
from PIL import Image

from .geometry_interface import GeometryBackend, GeometryFrame
from ..geometry.camera_model import CameraModel
from ..utils.pillow_compat import RESAMPLING


OFFICIAL_OMEGA_COMMIT = "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6"
OFFICIAL_OMEGA_CHECKPOINT_SHA256 = "c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934"
OFFICIAL_SOURCE_SHA256 = {
    "vggt_omega/models/vggt_omega.py": "190b92a37bd1ce27459a8ca92ac5fee4257f37c5b081316dbf685bee06c1f8e7",
    "vggt_omega/utils/load_fn.py": "a8da157a4c9fbf4212d70dbb9d065389d7beb41a9c6e23d2814f3bdee6d5f735",
    "vggt_omega/utils/pose_enc.py": "494b1b2482c748bc07499f5e695e0d36fd36dda4380d5a112b5e1b0937d31359",
    "vggt_omega/models/heads/dense_head.py": "40261c8ae7f12aca04f1ae0ff1dc6e2ae10d09daf2c6ecd781bbd3e2aad123d8",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PixelMapping:
    """Exact original-image to Omega-pixel mapping for a preprocessed view."""

    original_width: int
    original_height: int
    crop_left: int
    crop_top: int
    crop_width: int
    crop_height: int
    resize_width: int
    resize_height: int
    pad_left: int
    pad_top: int
    network_width: int
    network_height: int

    @property
    def matrix(self) -> np.ndarray:
        sx = self.resize_width / self.crop_width
        sy = self.resize_height / self.crop_height
        # PIL resize uses half-pixel centers.  Keeping this correction makes
        # detector masks and pointmaps agree at the image border.
        return np.array(
            [
                [sx, 0.0, self.pad_left - sx * self.crop_left + (sx - 1.0) / 2.0],
                [0.0, sy, self.pad_top - sy * self.crop_top + (sy - 1.0) / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "image_to_model": self.matrix.tolist(),
            "model_to_image": np.linalg.inv(self.matrix).tolist(),
            "coordinate_convention": "RGB image pixel centers to Omega model pixel centers",
        }


def _crop_to_supported_aspect_ratio(image: Image.Image, min_aspect: float = 0.5,
                                    max_aspect: float = 2.0) -> tuple[Image.Image, tuple[int, int]]:
    width, height = image.size
    ratio = height / max(width, 1)
    if ratio < min_aspect:
        crop_width = min(width, max(1, int(round(height / min_aspect))))
        left = max((width - crop_width) // 2, 0)
        return image.crop((left, 0, left + crop_width, height)), (left, 0)
    if ratio > max_aspect:
        crop_height = min(height, max(1, int(round(width * max_aspect))))
        top = max((height - crop_height) // 2, 0)
        return image.crop((0, top, width, top + crop_height)), (0, top)
    return image, (0, 0)


def _balanced_target_shape(aspect_ratio: float, resolution: int, patch_size: int = 16) -> tuple[int, int]:
    token_count = (resolution // patch_size) ** 2
    w_patches = max(1, int(np.round(np.sqrt(token_count / aspect_ratio))))
    h_patches = max(1, int(np.round(token_count / w_patches)))
    return h_patches * patch_size, w_patches * patch_size


def _preprocess_full_pair(images: Mapping[int, np.ndarray], resolution: int = 512):
    import torch
    from torchvision import transforms as TF

    views = sorted(int(v) for v in images)
    if len(views) != 2:
        raise ValueError(f"VGGT-Omega requires exactly two synchronized views, got {views}")
    tensors = []
    records = []
    for view in views:
        arr = np.asarray(images[view])
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"view {view} is not a complete HxWx3 RGB image: {arr.shape}")
        image = Image.fromarray(arr.astype(np.uint8), mode="RGB")
        original_w, original_h = image.size
        cropped, (crop_left, crop_top) = _crop_to_supported_aspect_ratio(image)
        crop_w, crop_h = cropped.size
        target_h, target_w = _balanced_target_shape(crop_h / max(crop_w, 1), resolution, 16)
        resized = cropped.resize((target_w, target_h), RESAMPLING.BICUBIC)
        tensors.append(TF.ToTensor()(resized))
        records.append({
            "view": view,
            "original_width": original_w,
            "original_height": original_h,
            "crop_left": crop_left,
            "crop_top": crop_top,
            "crop_width": crop_w,
            "crop_height": crop_h,
            "resize_width": target_w,
            "resize_height": target_h,
        })
    network_h = max(int(x.shape[1]) for x in tensors)
    network_w = max(int(x.shape[2]) for x in tensors)
    padded = []
    mappings = {}
    for tensor, rec in zip(tensors, records):
        h_pad, w_pad = network_h - tensor.shape[1], network_w - tensor.shape[2]
        top, left = h_pad // 2, w_pad // 2
        if h_pad or w_pad:
            tensor = torch.nn.functional.pad(tensor, (left, w_pad - left, top, h_pad - top), value=1.0)
        padded.append(tensor)
        mappings[int(rec["view"])] = PixelMapping(
            original_width=int(rec["original_width"]),
            original_height=int(rec["original_height"]),
            crop_left=int(rec["crop_left"]), crop_top=int(rec["crop_top"]),
            crop_width=int(rec["crop_width"]), crop_height=int(rec["crop_height"]),
            resize_width=int(rec["resize_width"]), resize_height=int(rec["resize_height"]),
            pad_left=int(left), pad_top=int(top),
            network_width=network_w, network_height=network_h,
        )
    return torch.stack(padded), mappings


def _unproject_depth(depth: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise ValueError(f"expected N,H,W depth, got {depth.shape}")
    n, h, w = depth.shape
    y, x = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")
    x = np.broadcast_to(x[None], (n, h, w))
    y = np.broadcast_to(y[None], (n, h, w))
    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]
    camera_points = np.stack([(x - cx) / fx * depth, (y - cy) / fy * depth, depth], axis=-1)
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    # Official Omega encodes camera-from-world extrinsics.
    return np.einsum("sij,shwj->shwi", np.transpose(rotation, (0, 2, 1)),
                     camera_points - translation[:, None, None, :])


class OmegaGeometryBackend(GeometryBackend):
    """Pinned official VGGT-Ω 512 inference, with no random fallback."""

    def __init__(self, source_root: str | Path, checkpoint: str | Path,
                 device: str = "cuda:0", image_resolution: int = 512,
                 verify_hash: bool = True, expose_intermediate_features: bool = False):
        self.source_root = Path(source_root).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.device_name = str(device)
        self.image_resolution = int(image_resolution)
        self.verify_hash = bool(verify_hash)
        self.expose_intermediate_features = bool(expose_intermediate_features)
        self.status = "NOT_INITIALIZED"
        self.reason = ""
        self.model = None
        self.torch = None
        self.decode_camera = None
        self.load_seconds = None
        self.checkpoint_sha256 = None
        self.source_commit = None
        self.last_predictions = None
        self.manifest = self._load_model()

    def _load_model(self) -> dict:
        manifest = {
            "status": self.status,
            "source_root": str(self.source_root),
            "checkpoint": str(self.checkpoint),
            "expected_source_commit": OFFICIAL_OMEGA_COMMIT,
            "expected_checkpoint_sha256": OFFICIAL_OMEGA_CHECKPOINT_SHA256,
            "image_resolution": self.image_resolution,
            "device": self.device_name,
            "expose_intermediate_features": self.expose_intermediate_features,
        }
        if self.image_resolution != 512:
            self.status, self.reason = "UNREGISTERED_RESOLUTION", "only the verified 512 checkpoint is registered"
            manifest.update(status=self.status, reason=self.reason)
            return manifest
        if not (self.source_root / "vggt_omega").is_dir():
            self.status, self.reason = "SOURCE_MISSING", str(self.source_root)
            manifest.update(status=self.status, reason=self.reason)
            return manifest
        if not self.checkpoint.is_file():
            self.status, self.reason = "WEIGHT_MISSING", str(self.checkpoint)
            manifest.update(status=self.status, reason=self.reason)
            return manifest
        try:
            self.checkpoint_sha256 = sha256_file(self.checkpoint)
            manifest["checkpoint_sha256"] = self.checkpoint_sha256
            if self.verify_hash and self.checkpoint_sha256 != OFFICIAL_OMEGA_CHECKPOINT_SHA256:
                raise ValueError("checkpoint SHA256 does not match the verified VGGT-Omega 512 bytes")
            for relative, expected in OFFICIAL_SOURCE_SHA256.items():
                actual = sha256_file(self.source_root / relative)
                if actual != expected:
                    raise ValueError(f"official Omega source hash mismatch: {relative}")
            import torch
            self.torch = torch
            if self.device_name.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            # Insert only the pinned official source.  The adapter does not
            # import any tracker or GMT package.
            source_str = str(self.source_root)
            if source_str not in sys.path:
                sys.path.insert(0, source_str)
            from vggt_omega.models import VGGTOmega
            from vggt_omega.utils.pose_enc import encoding_to_camera
            self.decode_camera = encoding_to_camera
            started = time.perf_counter()
            model = VGGTOmega(autocast=False).eval().requires_grad_(False)
            state = torch.load(str(self.checkpoint), map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            del state
            self.model = model.to(self.device_name)
            if self.device_name.startswith("cuda"):
                torch.cuda.synchronize(self.device_name)
            self.load_seconds = float(time.perf_counter() - started)
            self.source_commit = OFFICIAL_OMEGA_COMMIT
            self.status, self.reason = "READY", ""
        except Exception as exc:
            self.model = None
            self.status, self.reason = "DEPENDENCY_OR_INIT_FAILED", f"{type(exc).__name__}: {exc}"
        manifest.update(status=self.status, reason=self.reason,
                        checkpoint_sha256=self.checkpoint_sha256,
                        source_commit=self.source_commit,
                        load_seconds=self.load_seconds)
        return manifest

    def diagnostics(self) -> dict:
        return dict(self.manifest)

    @staticmethod
    def _as_numpy(value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        return np.asarray(value)

    def reconstruct(self, images: Mapping[int, np.ndarray], timestamp: int) -> GeometryFrame:
        if self.model is None:
            raise RuntimeError(f"VGGT-Omega backend is unavailable: {self.status}: {self.reason}")
        started = time.perf_counter()
        tensors, mappings = _preprocess_full_pair(images, self.image_resolution)
        t_pre = time.perf_counter()
        torch = self.torch
        with torch.inference_mode():
            # V100 has no native BF16 tensor cores; use FP16 for the measured
            # path while leaving the official FP32 depth/camera heads intact.
            amp_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            context = torch.autocast("cuda", dtype=amp_dtype) if str(self.device_name).startswith("cuda") else __import__("contextlib").nullcontext()
            with context:
                model_input = tensors.to(self.device_name, non_blocking=True)
                if self.expose_intermediate_features:
                    # Mirror the pinned VGGTOmega.forward path while retaining
                    # the final Aggregator patch tokens for object-level ROI
                    # features.  The official backbone weights remain frozen.
                    aggregator_input = model_input.unsqueeze(0) if model_input.ndim == 4 else model_input
                    aggregated, patch_start = self.model.aggregator(aggregator_input)
                    final_tokens = aggregated[-1]
                    predictions = {"camera_and_register_tokens": final_tokens[:, :, :patch_start].contiguous(),
                                   "images": aggregator_input, "aggregated_tokens_list": aggregated,
                                   "patch_token_start": patch_start}
                    with torch.autocast(device_type="cuda", enabled=False):
                        predictions["pose_enc"] = self.model.camera_head(aggregated, patch_token_start=patch_start)
                        predictions["depth"], predictions["depth_conf"] = self.model.dense_head(
                            aggregated, images=aggregator_input, patch_token_start=patch_start)
                else:
                    predictions = self.model(model_input)
            extrinsic, intrinsic = self.decode_camera(
                predictions["pose_enc"], predictions["images"].shape[-2:]
            )
        if str(self.device_name).startswith("cuda"):
            torch.cuda.synchronize(self.device_name)
        t_forward = time.perf_counter()
        depth = self._as_numpy(predictions["depth"])[0]
        depth_conf = self._as_numpy(predictions["depth_conf"])[0]
        extrinsic_np = self._as_numpy(extrinsic)[0]
        intrinsic_np = self._as_numpy(intrinsic)[0]
        points = _unproject_depth(depth, extrinsic_np, intrinsic_np)
        pointmaps, confidences, cameras = {}, {}, {}
        omega_patch_features = {}
        feature_status = "NOT_REQUESTED"
        if self.expose_intermediate_features:
            final_tokens = predictions["aggregated_tokens_list"][-1]
            patch_start = int(predictions["patch_token_start"])
            token_np = self._as_numpy(final_tokens)[0]
            patch_count = token_np.shape[1] - patch_start
            for idx, view in enumerate(sorted(int(v) for v in images)):
                hp, wp = points[idx].shape[:2]
                patch_h, patch_w = max(1, hp // 16), max(1, wp // 16)
                expected = patch_h * patch_w
                if patch_count != expected:
                    feature_status = "OMEGA_TOKEN_GRID_MISMATCH"
                    continue
                omega_patch_features[view] = token_np[idx, patch_start:].reshape(patch_h, patch_w, -1).astype(np.float32)
            if len(omega_patch_features) == len(images):
                feature_status = "OMEGA_FINAL_PATCH_FEATURES"
        views = sorted(int(v) for v in images)
        for idx, view in enumerate(views):
            mapping = mappings[view]
            h, w = points[idx].shape[:2]
            world_from_camera = np.eye(4, dtype=np.float64)
            world_from_camera[:3, :4] = np.linalg.inv(
                np.vstack([extrinsic_np[idx], [0.0, 0.0, 0.0, 1.0]])
            )[:3, :4]
            cameras[view] = CameraModel(
                intrinsic_model=intrinsic_np[idx],
                world_from_camera=world_from_camera,
                original_size=np.asarray(images[view]).shape[:2],
                model_size=(h, w),
                image_to_model=mapping.matrix,
                name=str(view),
            )
            pointmaps[view] = points[idx].astype(np.float64, copy=False)
            confidences[view] = depth_conf[idx].astype(np.float64, copy=False)
        finite = all(np.isfinite(x).all() for x in pointmaps.values())
        self.last_predictions = {
            "pose_enc": self._as_numpy(predictions["pose_enc"])[0],
            "depth": depth,
            "depth_conf": depth_conf,
            "extrinsic": extrinsic_np,
            "intrinsic": intrinsic_np,
            "omega_patch_features": {str(v): list(x.shape) for v, x in omega_patch_features.items()},
        }
        diagnostics = {
            "preprocess_s": float(t_pre - started),
            "model_only_s": float(t_forward - t_pre),
            "pointmap_postprocess_s": float(time.perf_counter() - t_forward),
            "total_s": float(time.perf_counter() - started),
            "input_views": views,
            "input_was_full_rgb": True,
            "model_shapes": [list(x.shape) for x in points],
            "depth_conf_median": {str(v): float(np.nanmedian(confidences[v])) for v in views},
            "checkpoint_sha256": self.checkpoint_sha256,
            "source_commit": self.source_commit,
            "mappings": {str(v): mappings[v].to_dict() for v in views},
            "finite": bool(finite),
        }
        return GeometryFrame(
            timestamp=int(timestamp), images=images, pointmaps=pointmaps,
            confidences=confidences, panoptic={}, query_features=np.empty((0, 0), np.float32),
            cameras=cameras,
            image_to_model={v: mappings[v].matrix for v in views},
            model_to_image={v: np.linalg.inv(mappings[v].matrix) for v in views},
            omega_patch_features=omega_patch_features,
            feature_status=feature_status,
            status="OK" if finite else "INVALID_WORLD_GEOMETRY", diagnostics=diagnostics,
        )


def map_mask_to_model(mask: np.ndarray, mapping: PixelMapping) -> np.ndarray:
    """Apply Omega's crop/resize/pad to an original-image boolean mask."""
    mask = np.asarray(mask).astype(np.uint8)
    cropped = mask[
        mapping.crop_top:mapping.crop_top + mapping.crop_height,
        mapping.crop_left:mapping.crop_left + mapping.crop_width,
    ]
    resized = np.asarray(
        Image.fromarray(cropped * 255).resize(
            (mapping.resize_width, mapping.resize_height), RESAMPLING.NEAREST
        )
    ) > 127
    out = np.zeros((mapping.network_height, mapping.network_width), dtype=bool)
    out[mapping.pad_top:mapping.pad_top + mapping.resize_height,
        mapping.pad_left:mapping.pad_left + mapping.resize_width] = resized
    return out

