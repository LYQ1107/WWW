"""Multi-camera causal static-background Sim(3) alignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import copy
from typing import Iterable

import numpy as np

from .world_alignment import AlignmentResult, Sim3, estimate_sim3
from ..runtime.online_worldmot import OnlineWorldMOT


@dataclass
class DualAlignmentResult:
    result: AlignmentResult
    per_camera: dict[int, AlignmentResult] = field(default_factory=dict)
    roundtrip_error: float = float("nan")
    used_cameras: tuple[int, ...] = ()
    rejected_cameras: dict[int, str] = field(default_factory=dict)


class DualCameraStaticAligner:
    """Estimate one transform from static correspondences in both views.

    Previous world pointmaps are supplied by the caller, so the implementation
    remains causal and does not read any GT or future frame.
    """

    def __init__(self, threshold: float = 0.08, min_inliers: int = 6, seed: int = 0):
        self.threshold = float(threshold)
        self.min_inliers = int(min_inliers)
        self.seed = int(seed)
        self.transform = Sim3()
        self.initialized = False

    def reset(self) -> None:
        self.transform = Sim3(); self.initialized = False

    def initialize(self) -> DualAlignmentResult:
        self.transform = Sim3(); self.initialized = True
        r = AlignmentResult(True, self.transform, 0, 0, 1.0, 0.0, 0.0, "initial_reference")
        return DualAlignmentResult(r, used_cameras=())

    def update(self, previous, current, cameras: Iterable[int] = (1, 2)) -> DualAlignmentResult:
        previous_world = copy(previous)
        previous_world.pointmaps = getattr(previous, "world_pointmaps", previous.pointmaps)
        src_all, dst_all = [], []
        per, rejected, used = {}, {}, []
        for view in sorted(set(int(v) for v in cameras)):
            if view not in previous.pointmaps or view not in current.pointmaps:
                rejected[view] = "missing_view"; continue
            src, dst = OnlineWorldMOT._static_correspondences(previous_world, current, view)
            if len(src) < 3:
                rejected[view] = "too_few_static_matches"; continue
            local = estimate_sim3(src, dst, threshold=self.threshold,
                                  min_inliers=max(3, min(self.min_inliers, len(src))),
                                  seed=self.seed + view)
            per[view] = local
            if local.success:
                src_all.append(src); dst_all.append(dst); used.append(view)
            else:
                rejected[view] = local.reason
        if not src_all:
            r = AlignmentResult(False, self.transform, 0, 0, 0.0, float("inf"), float("inf"),
                                "no_reliable_camera_matches")
            return DualAlignmentResult(r, per, float("nan"), tuple(used), rejected)
        src = np.concatenate(src_all, axis=0); dst = np.concatenate(dst_all, axis=0)
        r = estimate_sim3(src, dst, threshold=self.threshold,
                          min_inliers=max(3, self.min_inliers), seed=self.seed)
        if r.success:
            self.transform = r.transform; self.initialized = True
        # Camera-independent round-trip is the algebraic transform inverse
        # error, not a claim of absolute metric accuracy.
        probe = src[: min(len(src), 256)]
        if r.success and len(probe):
            inv = Sim3(1.0 / r.transform.scale, r.transform.rotation.T,
                       -r.transform.rotation.T @ r.transform.translation / r.transform.scale)
            roundtrip = inv.apply(r.transform.apply(probe))
            rt = float(np.median(np.linalg.norm(roundtrip - probe, axis=1)))
        else:
            rt = float("nan")
        return DualAlignmentResult(r, per, rt, tuple(used), rejected)
