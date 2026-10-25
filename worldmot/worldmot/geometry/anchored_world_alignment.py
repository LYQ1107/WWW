"""Causal keyframe-anchored world alignment."""

from __future__ import annotations

from dataclasses import dataclass
from copy import copy
from typing import Optional

import numpy as np

from .dual_view_static_alignment import DualCameraStaticAligner, DualAlignmentResult
from .world_alignment import AlignmentResult, Sim3
from ..runtime.online_worldmot import OnlineWorldMOT


@dataclass
class AnchorRecord:
    timestamp: int
    pointmaps: dict[int, np.ndarray]
    images: dict[int, np.ndarray]
    confidences: dict[int, np.ndarray]
    image_to_model: dict[int, np.ndarray]
    transform: Sim3
    dynamic_masks: dict | None = None


class AnchoredWorldAligner:
    """Use a bounded causal bank of reliable static keyframes."""

    def __init__(self, max_anchors: int = 8, min_inliers: int = 6, threshold: float = 0.08):
        self.max_anchors = int(max_anchors)
        self.dual = DualCameraStaticAligner(threshold=threshold, min_inliers=min_inliers)
        self.anchors: list[AnchorRecord] = []
        self.current_transform = Sim3()

    def reset(self) -> None:
        self.anchors.clear(); self.current_transform = Sim3(); self.dual.reset()

    def initialize(self, frame, timestamp: int) -> DualAlignmentResult:
        self.current_transform = Sim3()
        self.anchors = [AnchorRecord(int(timestamp), dict(frame.pointmaps), dict(frame.images),
                                     dict(frame.confidences), dict(frame.image_to_model), Sim3(),
                                     dict(getattr(frame,"dynamic_masks",{})))]
        return self.dual.initialize()

    def update(self, frame, timestamp: int) -> DualAlignmentResult:
        if not self.anchors:
            return self.initialize(frame, timestamp)
        best: Optional[DualAlignmentResult] = None
        for anchor in reversed(self.anchors):
            old = copy(frame)
            old.pointmaps = anchor.pointmaps; old.images = anchor.images
            old.confidences = anchor.confidences; old.image_to_model = anchor.image_to_model
            old.dynamic_masks = anchor.dynamic_masks or {}
            old.world_pointmaps = {v: anchor.transform.apply(p.reshape(-1, 3)).reshape(p.shape)
                                   for v, p in anchor.pointmaps.items()}
            result = self.dual.update(old, frame)
            if result.result.success:
                best = result; break
        if best is None:
            return DualAlignmentResult(AlignmentResult(False, self.current_transform, 0, 0, 0.0,
                                                        float("inf"), float("inf"), "anchor_rejected"))
        self.current_transform = best.result.transform
        if best.result.inlier_ratio >= 0.35 and best.result.residual_p95 <= 0.08:
            self.anchors.append(AnchorRecord(int(timestamp), dict(frame.pointmaps), dict(frame.images),
                                              dict(frame.confidences), dict(frame.image_to_model),
                                              self.current_transform,dict(getattr(frame,"dynamic_masks",{}))))
            self.anchors = self.anchors[-self.max_anchors:]
        return best
