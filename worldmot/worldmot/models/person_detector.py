"""Independent COCO person detection for the WorldMOT ablation.

The adapter is deliberately small and explicit about its dependency boundary.
It uses the official MMDetection RTMDet-Ins-tiny COCO checkpoint when the
caller supplies a local checkpoint.  A missing OpenMMLab installation or
checkpoint is reported as a blocked detector; no boxes or masks are invented.
The same class can load the bbox-only RTMDet-tiny config for a registered
speed/quality control.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import time
from typing import Mapping, Optional

import numpy as np


RTMDET_INS_TINY_CONFIG = (
    "https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/rtmdet/"
    "rtmdet-ins_tiny_8xb32-300e_coco.py"
)
RTMDET_INS_TINY_CHECKPOINT = (
    "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
    "rtmdet-ins_tiny_8xb32-300e_coco/"
    "rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth"
)
RTMDET_TINY_CONFIG = (
    "https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/rtmdet/"
    "rtmdet_tiny_8xb32-300e_coco.py"
)
RTMDET_TINY_CHECKPOINT = (
    "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
    "rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
)


def sha256_file(path: str | Path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class PersonDetection:
    view_id: int
    detection_id: int
    bbox_xywh: tuple[float, float, float, float]
    score: float
    mask: np.ndarray | None = None
    label: int = 0
    source: str = "mmdet-rtmdet-ins-tiny-coco"

    @property
    def area(self) -> float:
        return float(max(0.0, self.bbox_xywh[2]) * max(0.0, self.bbox_xywh[3]))

    def to_dict(self, include_mask: bool = False) -> dict:
        out = {
            "view_id": int(self.view_id),
            "detection_id": int(self.detection_id),
            "bbox_xywh": [float(x) for x in self.bbox_xywh],
            "score": float(self.score),
            "label": int(self.label),
            "source": self.source,
        }
        # Persist a compact, lossless mask representation with every saved
        # detection.  The full HxW matrix is intentionally avoided because a
        # 100-frame run would otherwise become needlessly large; ``rle_c`` is
        # a simple row-major run-length encoding that can be decoded without
        # pycocotools.  ``include_mask`` remains as a compatibility option for
        # callers that explicitly need the dense matrix.
        if self.mask is not None:
            mask = np.asarray(self.mask, dtype=bool)
            flat = mask.reshape(-1).astype(np.uint8)
            padded = np.concatenate(([0], flat, [0]))
            changes = np.flatnonzero(padded[1:] != padded[:-1])
            counts = np.diff(changes).astype(int).tolist()
            out["mask_encoding"] = "rle_c"
            out["mask_shape"] = [int(mask.shape[0]), int(mask.shape[1])]
            out["mask_counts"] = counts
            out["mask_area"] = int(mask.sum())
        else:
            out["mask_encoding"] = None
            out["mask_shape"] = None
            out["mask_counts"] = None
            out["mask_area"] = 0
        if include_mask:
            out["mask"] = self.mask.astype(np.uint8).tolist() if self.mask is not None else None
        return out


class PersonDetector:
    """MMDetection RTMDet adapter with auditable status and timings."""

    def __init__(
        self,
        config: str | Path,
        checkpoint: str | Path,
        device: str = "cuda:0",
        score_threshold: float = 0.25,
        max_detections: int = 100,
        backend_name: str = "rtmdet-ins-tiny",
        expose_intermediate_features: bool = False,
    ) -> None:
        self.config = str(Path(config).resolve())
        self.checkpoint = str(Path(checkpoint).resolve())
        self.device = str(device)
        self.score_threshold = float(score_threshold)
        self.max_detections = int(max_detections)
        self.backend_name = str(backend_name)
        self.expose_intermediate_features = bool(expose_intermediate_features)
        self.model = None
        self.status = "NOT_INITIALIZED"
        self.reason = ""
        self.model_version = None
        self.mmcv_version = None
        self.mmengine_version = None
        self.weight_sha256 = sha256_file(self.checkpoint)
        self.initialization_seconds = None
        self.last_timings: dict[str, float] = {}
        self.last_feature_maps: dict[int, np.ndarray] = {}
        self._feature_hook = None
        self._last_backbone_output = None
        self._init_model()

    def _init_model(self) -> None:
        if not Path(self.config).is_file():
            self.status, self.reason = "CONFIG_MISSING", f"config not found: {self.config}"
            return
        if not Path(self.checkpoint).is_file():
            self.status, self.reason = "WEIGHT_MISSING", f"checkpoint not found: {self.checkpoint}"
            return
        started = time.perf_counter()
        try:
            import mmcv
            import mmengine
            import mmdet
            from mmdet.apis import init_detector
            self.mmcv_version = getattr(mmcv, "__version__", None)
            self.mmengine_version = getattr(mmengine, "__version__", None)
            self.model_version = getattr(mmdet, "__version__", None)
            self.model = init_detector(self.config, self.checkpoint, device=self.device)
            self.model.eval()
            if self.expose_intermediate_features and hasattr(self.model, "backbone"):
                def _capture(_module, _inputs, output):
                    self._last_backbone_output = output
                self._feature_hook = self.model.backbone.register_forward_hook(_capture)
            self.status = "READY"
            self.reason = ""
        except Exception as exc:  # dependency failures must be visible to reports
            self.model = None
            self.status = "DEPENDENCY_OR_INIT_FAILED"
            self.reason = f"{type(exc).__name__}: {exc}"
        self.initialization_seconds = float(time.perf_counter() - started)

    def manifest(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "backend": self.backend_name,
            "config": self.config,
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.weight_sha256,
            "device": self.device,
            "score_threshold": self.score_threshold,
            "max_detections": self.max_detections,
            "expose_intermediate_features": self.expose_intermediate_features,
            "mmdet_version": self.model_version,
            "mmcv_version": self.mmcv_version,
            "mmengine_version": self.mmengine_version,
            "initialization_seconds": self.initialization_seconds,
        }

    @staticmethod
    def _array(x) -> np.ndarray:
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        return np.asarray(x)

    def infer(self, images: Mapping[int, np.ndarray]) -> dict[int, list[PersonDetection]]:
        """Run detector on each complete RGB image independently."""
        if self.model is None:
            self.last_timings = {str(v): 0.0 for v in images}
            return {int(v): [] for v in images}
        from mmdet.apis import inference_detector

        outputs: dict[int, list[PersonDetection]] = {}
        self.last_timings = {}
        self.last_feature_maps = {}
        torch = None
        if self.device.startswith("cuda"):
            import torch as _torch
            torch = _torch
        for view, image in sorted(images.items()):
            if torch is not None:
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            self._last_backbone_output = None
            result = inference_detector(self.model, np.asarray(image))
            if torch is not None:
                torch.cuda.synchronize(self.device)
            # MMDetection 3.x returns DetDataSample.pred_instances.  Keep a
            # strict branch so a future API change becomes a visible failure.
            pred = getattr(result, "pred_instances", None)
            if pred is None:
                raise RuntimeError("MMDetection result has no pred_instances")
            boxes = self._array(getattr(pred, "bboxes", np.empty((0, 4))))
            scores = self._array(getattr(pred, "scores", np.empty((0,))))
            labels = self._array(getattr(pred, "labels", np.empty((0,), dtype=np.int64)))
            masks = getattr(pred, "masks", None)
            masks = self._array(masks) if masks is not None else None
            detections: list[PersonDetection] = []
            order = np.argsort(-scores.reshape(-1)) if len(scores) else np.empty((0,), dtype=np.int64)
            for raw_idx in order[: self.max_detections]:
                idx = int(raw_idx)
                label = int(labels[idx]) if idx < len(labels) else -1
                score = float(scores[idx]) if idx < len(scores) else 0.0
                # COCO's person category is label 0.  Do not turn other COCO
                # objects into people just because they have a mask.
                if label != 0 or score < self.score_threshold:
                    continue
                if idx >= len(boxes):
                    continue
                x0, y0, x1, y1 = [float(x) for x in boxes[idx, :4]]
                mask = None
                if masks is not None and idx < len(masks):
                    mask = np.asarray(masks[idx]).astype(bool)
                    if mask.shape != np.asarray(image).shape[:2]:
                        from PIL import Image
                        mask = np.asarray(
                            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                                (np.asarray(image).shape[1], np.asarray(image).shape[0]),
                                Image.Resampling.NEAREST,
                            )
                        ) > 127
                detections.append(
                    PersonDetection(
                        view_id=int(view), detection_id=len(detections),
                        bbox_xywh=(x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)),
                        score=score, mask=mask, source=f"mmdet-{self.backend_name}-coco",
                    )
                )
            outputs[int(view)] = detections
            if self.expose_intermediate_features and self._last_backbone_output is not None:
                feature = self._last_backbone_output
                if isinstance(feature, dict):
                    feature = next(reversed(feature.values()))
                elif isinstance(feature, (tuple, list)):
                    feature = feature[-1]
                feature = self._array(feature)
                if feature.ndim == 4:
                    feature = feature[0]
                if feature.ndim == 3:
                    self.last_feature_maps[int(view)] = feature.astype(np.float32, copy=False)
            self.last_timings[str(int(view))] = float(time.perf_counter() - started)
        return outputs


# Backward-compatible name for callers that imported the old placeholder.
PersonFallback = PersonDetector

