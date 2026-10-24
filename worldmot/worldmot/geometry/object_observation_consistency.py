"""Consistency tests for multi-view observations of one world object."""

from __future__ import annotations

import numpy as np

from ..models.world_object_support import WorldObjectSupport, support_compatibility


def observation_consistency(left: WorldObjectSupport, right: WorldObjectSupport,
                            max_center_distance: float = 3.5) -> tuple[bool, dict]:
    score, detail, reason = support_compatibility(left, right, max_center_distance)
    detail = dict(detail)
    detail.update({"score": float(score), "reason": reason,
                   "left_ambiguity": left.ambiguity, "right_ambiguity": right.ambiguity})
    return bool(np.isfinite(score)), detail


def merge_supports(supports: list[WorldObjectSupport]) -> WorldObjectSupport:
    if not supports:
        raise ValueError("cannot merge no supports")
    box = np.concatenate([x.box_points for x in supports if len(x.box_points)], axis=0) if any(len(x.box_points) for x in supports) else np.empty((0, 3))
    fg = np.concatenate([x.foreground_points for x in supports if len(x.foreground_points)], axis=0) if any(len(x.foreground_points) for x in supports) else np.empty((0, 3))
    from ..models.world_object_support import _stats
    center, cov = _stats(fg)
    return WorldObjectSupport(box, fg, center, cov,
                              float(np.mean([x.quality for x in supports])),
                              float(np.mean([x.foreground_fraction for x in supports])),
                              "MERGED" if all(x.ambiguity == "DEPTH_CLUSTERED" for x in supports) else "MIXED",
                              float(np.mean([x.confidence_median for x in supports])),
                              float(np.mean([x.visible_fraction for x in supports])),
                              {"source_count": len(supports)})
