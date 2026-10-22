"""Leakage-safe identity and visibility targets from detector/GT overlap."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Mapping, Sequence

import numpy as np


def iou_xywh(a, b) -> float:
    ax, ay, aw, ah = [float(x) for x in a]; bx, by, bw, bh = [float(x) for x in b]
    x0, y0 = max(ax, bx), max(ay, by); x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = max(aw, 0.0) * max(ah, 0.0) + max(bw, 0.0) * max(bh, 0.0) - inter
    return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class IdentityTarget:
    token_index: int
    camera_id: int
    identity_key: str | None
    identity_class: int
    visible: bool
    matched_iou: float
    source: str


def match_detections_to_gt(detections: Mapping[int, Sequence], ground_truth: Mapping[int, Sequence[dict]],
                          scene_id: int, iou_threshold: float = 0.5) -> dict[tuple[int, int], IdentityTarget]:
    """Match only for supervision/evaluation; runtime never calls this function."""
    out = {}
    for view, ds in detections.items():
        gt = list(ground_truth.get(int(view), []))
        keys = [a.get("instance_id", a.get("track_id", a.get("id"))) for a in gt]
        ambiguous = {key for key, count in Counter(keys).items() if count > 1}
        pairs = []
        for di, det in enumerate(ds):
            for gi, ann in enumerate(gt):
                pairs.append((iou_xywh(det.bbox_xywh, ann.get("bbox", [0, 0, 0, 0])), di, gi))
        used_d, used_g = set(), set()
        for score, di, gi in sorted(pairs, reverse=True):
            if score < iou_threshold or di in used_d or gi in used_g:
                continue
            used_d.add(di); used_g.add(gi)
            ann = gt[gi]
            raw = ann.get("instance_id", ann.get("track_id", ann.get("id")))
            if raw in ambiguous:
                out[(int(view), int(di))] = IdentityTarget(-1, int(view), None, -1, False, float(score), "ambiguous_duplicate_identity_masked")
                continue
            key = f"scene{int(scene_id)}:id{int(raw)}" if raw is not None else None
            out[(int(view), int(di))] = IdentityTarget(-1, int(view), key, -1, True, float(score), "bbox_iou_gt_supervision")
        for di in range(len(ds)):
            out.setdefault((int(view), di), IdentityTarget(-1, int(view), None, -1, False, 0.0, "unmatched_detector"))
    return out


def assign_relative_identity_classes(targets: Sequence[IdentityTarget], seed: int = 0) -> list[IdentityTarget]:
    """Map scene-qualified IDs to a clip-local randomized class ordering."""
    keys = sorted({t.identity_key for t in targets if t.identity_key is not None})
    rng = np.random.default_rng(int(seed))
    perm = list(rng.permutation(len(keys))) if keys else []
    class_by_key = {key: int(perm[i]) for i, key in enumerate(keys)}
    return [IdentityTarget(t.token_index, t.camera_id, t.identity_key,
                           class_by_key.get(t.identity_key, -1), t.visible,
                           t.matched_iou, t.source) for t in targets]
