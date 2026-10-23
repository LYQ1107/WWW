"""Object-level visual features for WorldMOT v2.

The v1 identity feature ``[score, number_of_views]`` is intentionally not
used here.  The extractor accepts real RGB crops and, when the backends expose
them, ROI-pooled RTMDet and VGGT-Omega intermediate features.  Missing learned
feature maps are reported explicitly rather than silently replaced by a
claimed ReID embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .person_detector import PersonDetection
from ..utils.pillow_compat import RESAMPLING


@dataclass(frozen=True)
class ObjectFeature:
    view_id: int
    detection_id: int
    appearance_feature: np.ndarray
    detector_feature: np.ndarray
    omega_feature: np.ndarray
    source: str
    valid: bool
    diagnostics: dict
    rgb_feature: np.ndarray | None = None


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(x))
    return x / max(norm, 1e-8)


def _fixed_pool(x: np.ndarray, dim: int) -> np.ndarray:
    """Compress a variable ROI tensor without copying a single patch.

    Channel/spatial source maps have backend-dependent dimensions.  Equal
    contiguous bins provide a deterministic fixed-size source feature while
    retaining all ROI values; the trainable query encoder performs the learned
    identity projection later.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if not len(x):
        return np.zeros(int(dim), dtype=np.float32)
    bins = np.array_split(x, int(dim))
    return _normalize(np.asarray([part.mean() if len(part) else 0.0 for part in bins], dtype=np.float32))


def _resize_crop(image: np.ndarray, bbox: Sequence[float], size: int = 16) -> np.ndarray:
    """Resize an RGB crop using PIL, keeping this module CPU-only and pinned."""

    from PIL import Image

    arr = np.asarray(image)
    h, w = arr.shape[:2]
    x, y, bw, bh = [float(v) for v in bbox]
    x0, y0 = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
    x1, y1 = min(w, int(np.ceil(x + bw))), min(h, int(np.ceil(y + bh)))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((size, size, 3), dtype=np.float32)
    crop = arr[y0:y1, x0:x1, :3].astype(np.uint8)
    return np.asarray(Image.fromarray(crop, mode="RGB").resize((size, size), RESAMPLING.BILINEAR), dtype=np.float32) / 255.0


def _rgb_descriptor(image: np.ndarray, bbox: Sequence[float], mask: np.ndarray | None) -> np.ndarray:
    crop = _resize_crop(image, bbox, 16)
    # Real RGB appearance statistics: spatial pooled RGB, first derivatives,
    # and channel moments.  This is a source descriptor, not a pretrained
    # identity embedding; the trainable observation encoder learns its map.
    pooled = crop.reshape(4, 4, 4, 4, 3).mean(axis=(1, 3)).reshape(-1)
    dx = np.diff(crop, axis=1, prepend=crop[:, :1]).mean(axis=2).reshape(-1)
    dy = np.diff(crop, axis=0, prepend=crop[:1]).mean(axis=2).reshape(-1)
    moments = np.concatenate([crop.mean((0, 1)), crop.std((0, 1)), np.quantile(crop.reshape(-1, 3), [0.1, 0.5, 0.9], axis=0).reshape(-1)])
    out = np.concatenate([pooled, dx, dy, moments]).astype(np.float32)
    return _normalize(out)


def _as_chw(feature_map: np.ndarray) -> np.ndarray:
    x = np.asarray(feature_map)
    if x.ndim == 4:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"feature map must have 3 dimensions, got {x.shape}")
    # C,H,W is the usual torch representation and has a channel dimension
    # larger than its spatial dimensions.  Omega caches are often H,W,C.
    if x.shape[0] >= x.shape[1] and x.shape[0] >= x.shape[2]:
        return x.astype(np.float32, copy=False)
    return np.moveaxis(x, -1, 0).astype(np.float32, copy=False)


def _roi_pool(feature_map: np.ndarray, bbox: Sequence[float], image_shape: tuple[int, int], output: int = 4) -> np.ndarray:
    fmap = _as_chw(feature_map)
    c, h, w = fmap.shape
    ih, iw = image_shape
    x, y, bw, bh = [float(v) for v in bbox]
    x0 = max(0, min(w - 1, int(np.floor(x / max(iw, 1) * w))))
    y0 = max(0, min(h - 1, int(np.floor(y / max(ih, 1) * h))))
    x1 = max(x0 + 1, min(w, int(np.ceil((x + bw) / max(iw, 1) * w))))
    y1 = max(y0 + 1, min(h, int(np.ceil((y + bh) / max(ih, 1) * h))))
    crop = fmap[:, y0:y1, x0:x1]
    # Adaptive pooling without requiring torchvision's compiled roi_align.
    yy = np.linspace(0, crop.shape[1], output + 1, dtype=np.int64)
    xx = np.linspace(0, crop.shape[2], output + 1, dtype=np.int64)
    pooled = np.empty((c, output, output), dtype=np.float32)
    for iy in range(output):
        for ix in range(output):
            cell = crop[:, yy[iy]:max(yy[iy + 1], yy[iy] + 1), xx[ix]:max(xx[ix + 1], xx[ix] + 1)]
            pooled[:, iy, ix] = cell.mean(axis=(1, 2))
    return _normalize(pooled)


class FrozenObjectFeatureExtractor:
    """Extract auditable source features without identity-label leakage.

    ``detector_feature_maps`` and ``omega_patch_features`` are optional maps
    emitted by the respective frozen backends.  The RGB descriptor is always
    available for valid crops, while formal v2 runs record whether each
    learned source was present.
    """

    def __init__(self, appearance_dim: int = 256, require_intermediate: bool = False):
        self.appearance_dim = int(appearance_dim)
        self.require_intermediate = bool(require_intermediate)

    def extract(
        self,
        detections: Mapping[int, Sequence[PersonDetection]],
        images: Mapping[int, np.ndarray],
        detector_feature_maps: Mapping[int, np.ndarray] | None = None,
        omega_patch_features: Mapping[int, np.ndarray] | None = None,
    ) -> dict[tuple[int, int], ObjectFeature]:
        detector_feature_maps = detector_feature_maps or {}
        omega_patch_features = omega_patch_features or {}
        out: dict[tuple[int, int], ObjectFeature] = {}
        for view, view_detections in sorted(detections.items()):
            image = np.asarray(images[int(view)])
            for det in view_detections:
                rgb = _fixed_pool(_rgb_descriptor(image, det.bbox_xywh, det.mask), 128)
                det_feat = np.zeros(0, dtype=np.float32)
                omega_feat = np.zeros(0, dtype=np.float32)
                native_roi = getattr(det, "detector_roi_feature", None)
                if native_roi is not None:
                    native_roi = np.asarray(native_roi, dtype=np.float32)
                    if not native_roi.size or not np.isfinite(native_roi).all():
                        raise ValueError("Invalid frozen detector ROI feature")
                    det_feat = _fixed_pool(native_roi, 128)
                elif int(view) in detector_feature_maps:
                    det_feat = _fixed_pool(
                        _roi_pool(detector_feature_maps[int(view)], det.bbox_xywh, image.shape[:2]), 128,
                    )
                if int(view) in omega_patch_features:
                    omega_feat = _fixed_pool(
                        _roi_pool(omega_patch_features[int(view)], det.bbox_xywh, image.shape[:2]), 256,
                    )
                # Each branch has a separate learned projection in PWQT.
                # Never concatenate then truncate: every source survives.
                merged = np.concatenate([rgb,
                    det_feat if len(det_feat) else np.zeros(128, np.float32),
                    omega_feat if len(omega_feat) else np.zeros(256, np.float32)])
                learned_ok = len(det_feat) > 0 and len(omega_feat) > 0
                is_gmt = det.source == "gmt_pre_gta_detector"
                detector_name = "gmt_roi" if is_gmt else "rtmdet_roi"
                source = f"rgb+{detector_name}+omega_patch" if learned_ok else "rgb_source_only"
                valid = len(rgb) > 0 and (learned_ok or not self.require_intermediate)
                out[(int(view), int(det.detection_id))] = ObjectFeature(
                    int(view), int(det.detection_id), merged.astype(np.float32), det_feat, omega_feat,
                    source, valid,
                    {"rgb_available": True, "detector_roi_available": bool(len(det_feat)),
                     "detector_roi_source": detector_name,
                     "rtmdet_roi_available": bool(len(det_feat)) and not is_gmt,
                     "omega_patch_available": bool(len(omega_feat)),
                     "formal_intermediate_required": self.require_intermediate}, rgb,
                )
        return out
