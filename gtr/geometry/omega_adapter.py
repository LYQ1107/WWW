"""Frozen official Omega interface, independent of the old feature branch.

Preprocessing always calls the pinned official implementation. Its helper
functions are reused to describe crop/resize/padding; no 518/14 mapping is used.
This module is not connected to GMT until the scientific gates are satisfied.
"""
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import sys
import time

import numpy as np

OFFICIAL_COMMIT = '48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6'
SOURCE_HASHES = {
    'vggt_omega/models/vggt_omega.py': '190b92a37bd1ce27459a8ca92ac5fee4257f37c5b081316dbf685bee06c1f8e7',
    'vggt_omega/utils/load_fn.py': 'a8da157a4c9fbf4212d70dbb9d065389d7beb41a9c6e23d2814f3bdee6d5f735',
    'vggt_omega/utils/pose_enc.py': '494b1b2482c748bc07499f5e695e0d36fd36dda4380d5a112b5e1b0937d31359',
    'vggt_omega/models/heads/dense_head.py': '40261c8ae7f12aca04f1ae0ff1dc6e2ae10d09daf2c6ecd781bbd3e2aad123d8',
}
OFFICIAL_WEIGHTS = {
    416: ('vggt_omega_1b_416_reproduce.pt', 'e900833330eb0e570e10543e6108d0dd704fa7c8dbc32738a645987875298896'),
    512: ('vggt_omega_1b_512.pt', 'c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934'),
}


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def official_api(source):
    source = Path(source).resolve()
    for relative, expected in SOURCE_HASHES.items():
        if file_sha(source / relative) != expected:
            raise ValueError(f'official source hash mismatch: {relative}')
    sys.path.insert(0, str(source))
    from vggt_omega.utils import load_fn
    from vggt_omega.utils.pose_enc import encoding_to_camera
    if Path(load_fn.__file__).resolve() != source / 'vggt_omega/utils/load_fn.py':
        raise ValueError('another Omega installation shadows the pinned source')
    return load_fn, encoding_to_camera


@dataclass(frozen=True)
class PixelMapping:
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

    def __post_init__(self):
        positive = ('original_width', 'original_height', 'crop_width', 'crop_height',
                    'resize_width', 'resize_height', 'network_width', 'network_height')
        if any(getattr(self, k) <= 0 for k in positive):
            raise ValueError('nonpositive image extent')
        if min(self.crop_left, self.crop_top, self.pad_left, self.pad_top) < 0:
            raise ValueError('negative crop/pad')
        if (self.crop_left + self.crop_width > self.original_width or
            self.crop_top + self.crop_height > self.original_height or
            self.pad_left + self.resize_width > self.network_width or
            self.pad_top + self.resize_height > self.network_height):
            raise ValueError('crop/pad exceeds image extent')

    @property
    def A_edges(self):
        sx, sy = self.resize_width / self.crop_width, self.resize_height / self.crop_height
        return np.array([[sx, 0., self.pad_left-sx*self.crop_left],
                         [0., sy, self.pad_top-sy*self.crop_top], [0., 0., 1.]])

    @property
    def A(self):
        # OpenCV integer pixel-center coordinates. PIL bicubic uses half-pixel
        # centers: x_new=(x_old-crop_left+.5)*sx-.5+pad_left.
        a = self.A_edges
        a[0, 2] += (a[0, 0]-1.)/2
        a[1, 2] += (a[1, 1]-1.)/2
        return a

    def forward(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        out = np.c_[points, np.ones(len(points))] @ self.A.T
        valid = np.isfinite(points).all(-1)
        valid &= (points[:, 0] >= self.crop_left-.5) & (points[:, 0] < self.crop_left+self.crop_width-.5)
        valid &= (points[:, 1] >= self.crop_top-.5) & (points[:, 1] < self.crop_top+self.crop_height-.5)
        return out[:, :2], valid

    def inverse(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        values = np.c_[points, np.ones(len(points))] @ np.linalg.inv(self.A).T
        valid = np.isfinite(points).all(-1)
        valid &= (points[:, 0] >= self.pad_left-.5) & (points[:, 0] < self.pad_left+self.resize_width-.5)
        valid &= (points[:, 1] >= self.pad_top-.5) & (points[:, 1] < self.pad_top+self.resize_height-.5)
        values[~valid, :2] = np.nan
        return values[:, :2], valid

    def record(self):
        return dict(**asdict(self), A_original_to_omega=self.A.tolist(),
                    A_original_edges_to_omega_edges=self.A_edges.tolist(),
                    A_omega_to_original=np.linalg.inv(self.A).tolist(),
                    coordinate_convention='OpenCV pixel centers; separate continuous edge matrix; cropped/padded points invalid')


def preprocess_pair(source, image_paths, resolution=416):
    if len(image_paths) != 2 or resolution not in OFFICIAL_WEIGHTS:
        raise ValueError('exactly two images and an explicitly registered resolution required')
    load_fn, _ = official_api(source)
    tensor = load_fn.load_and_preprocess_images(image_paths, image_resolution=resolution, mode='balanced')
    H, W = tensor.shape[-2:]
    mappings = []
    for path in image_paths:
        image = load_fn._load_rgb_image(path)
        ow, oh = image.size
        cropped = load_fn._crop_to_supported_aspect_ratio(image)
        cw, ch = cropped.size
        rh, rw = load_fn._balanced_target_shape(ch/cw, resolution, 16)
        mappings.append(PixelMapping(ow, oh, (ow-cw)//2, (oh-ch)//2, cw, ch, rw, rh,
                                     (W-rw)//2, (H-rh)//2, W, H))
    if tuple(tensor.shape) != (2, 3, H, W) or not np.isfinite(tensor.numpy()).all():
        raise ValueError('invalid official preprocessed tensor')
    return tensor, mappings


class OmegaAdapter:
    """Explicit checkpoint/hash load. No random-weight fallback; no online fusion."""
    def __init__(self, source, checkpoint, resolution=416, device='cuda:0'):
        import torch
        if resolution not in OFFICIAL_WEIGHTS:
            raise ValueError('unregistered checkpoint configuration')
        self.source, self.resolution, self.device = source, resolution, device
        self.load_started = time.perf_counter()
        if file_sha(checkpoint) != OFFICIAL_WEIGHTS[resolution][1]:
            raise ValueError('weight bytes do not match the official Hugging Face LFS SHA256')
        self.load_fn, self.decode = official_api(source)
        from vggt_omega.models import VGGTOmega
        self.model = VGGTOmega().eval().requires_grad_(False)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.state_keys = len(state)
        del state
        self.model = self.model.to(device)
        torch.cuda.synchronize(device)
        self.load_seconds = time.perf_counter()-self.load_started

    def infer_pair(self, image_paths, camera_ids, timestamp):
        import torch
        if len(camera_ids) != 2 or len(set(camera_ids)) != 2:
            raise ValueError('two distinct physical camera IDs required')
        started = time.perf_counter()
        try:
            images, mappings = preprocess_pair(self.source, image_paths, self.resolution)
            with torch.inference_mode():
                predictions = self.model(images.to(self.device))
                extrinsics, intrinsics = self.decode(predictions['pose_enc'], predictions['images'].shape[-2:])
            torch.cuda.synchronize(self.device)
            expected_shapes = {'pose_enc': (1, 2, 9), 'images': (1, 2, *images.shape[1:]),
                               'depth': (1, 2, *images.shape[-2:], 1), 'depth_conf': (1, 2, *images.shape[-2:])}
            for name, shape in expected_shapes.items():
                if tuple(predictions[name].shape) != shape or not torch.isfinite(predictions[name]).all().item():
                    raise ValueError(f'invalid {name} shape or nonfinite values')
            if tuple(extrinsics.shape) != (1, 2, 3, 4) or tuple(intrinsics.shape) != (1, 2, 3, 3):
                raise ValueError('invalid decoded camera shapes')
            if not torch.isfinite(extrinsics).all().item() or not torch.isfinite(intrinsics).all().item():
                raise ValueError('nonfinite decoded cameras')
            return dict(status='FINITE_INTERFACE_OUTPUTS', camera_ids=list(camera_ids), timestamp=timestamp,
                        resolution=self.resolution, load_seconds=self.load_seconds,
                        preprocess_and_forward_seconds=time.perf_counter()-started,
                        mappings=mappings, predictions=predictions,
                        extrinsics=extrinsics[0].cpu().numpy(), intrinsics=intrinsics[0].cpu().numpy(),
                        geometry_reliable='NOT_MEASURED',
                        depth_conf_semantics='1+exp(logit); uncalibrated, not identity probability')
        except (RuntimeError, ValueError, OSError) as exc:
            return dict(status='INVALID_GEOMETRY_INTERFACE', camera_ids=list(camera_ids), timestamp=timestamp,
                        exception_type=type(exc).__name__, reason=str(exc),
                        preprocess_and_forward_seconds=time.perf_counter()-started,
                        geometry_reliable=False)
