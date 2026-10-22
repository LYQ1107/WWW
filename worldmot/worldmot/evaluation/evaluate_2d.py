"""Post-hoc 2-D and identity diagnostics for a WorldMOT JSONL run."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from ..data.visiontrack import VisionTrack


def iou_xywh(a, b) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0., x1 - x0) * max(0., y1 - y0)
    union = max(aw, 0.) * max(ah, 0.) + max(bw, 0.) * max(bh, 0.) - inter
    return float(inter / union) if union > 0 else 0.0


def _match(preds: list[dict], anns: list[dict], threshold: float):
    pairs = []
    for pi, p in enumerate(preds):
        if not p.get("visible") or p.get("bbox_xywh") is None:
            continue
        for ai, a in enumerate(anns):
            score = iou_xywh(p["bbox_xywh"], a["bbox"])
            if score >= threshold:
                pairs.append((score, pi, ai))
    used_p, used_a, out = set(), set(), []
    for score, pi, ai in sorted(pairs, reverse=True):
        if pi in used_p or ai in used_a:
            continue
        used_p.add(pi); used_a.add(ai); out.append((pi, ai, score))
    return out


def evaluate_jsonl(dataset: VisionTrack, jsonl: str | Path, iou_threshold: float = 0.3) -> dict:
    rows = [json.loads(x) for x in Path(jsonl).read_text(encoding="utf-8").splitlines() if x.strip()]
    matched = 0; gt_total = 0; pred_visible = 0; switches = 0; fragments = 0
    prev_by_gt: dict[tuple[int, int, int], int] = {}
    seen_by_gt: dict[tuple[int, int, int], bool] = {}
    cross_total = 0; cross_equal = 0; cross_conflicts = []
    frame_records = []
    for row in rows:
        scene = int(row["scene_id"] or row.get("scene", 0)); frame = int(row["timestamp"])
        pair_gt = {}
        # The run stores image IDs so this remains post-hoc and does not alter inference.
        for view, image_id in row.get("image_ids", {}).items():
            pair_gt[int(view)] = dataset.ground_truth(int(image_id))
        matched_by_view = {}
        for view, preds in row.get("tracks_2d_by_camera", {}).items():
            view_i = int(view); anns = pair_gt.get(view_i, [])
            gt_total += len(anns)
            pred_visible += sum(1 for p in preds if p.get("visible") and p.get("bbox_xywh") is not None)
            pairs = _match(preds, anns, iou_threshold)
            matched += len(pairs)
            matched_by_view[view_i] = [(int(anns[ai].get("instance_id", -1)), int(preds[pi]["global_id"]), float(s)) for pi, ai, s in pairs]
            for gt_id, gid, _ in matched_by_view[view_i]:
                key = (scene, view_i, gt_id)
                if key in prev_by_gt and prev_by_gt[key] != gid:
                    switches += 1
                if seen_by_gt.get(key, False) is False and key in prev_by_gt:
                    fragments += 1
                prev_by_gt[key] = gid; seen_by_gt[key] = True
        # Same original VisionTrack instance ids across the synchronized views
        # are a diagnostic only; ambiguous/duplicate GT IDs are reported.
        if len(matched_by_view) >= 2:
            left = defaultdict(list); right = defaultdict(list)
            for gt, gid, _ in matched_by_view.get(1, []): left[gt].append(gid)
            for gt, gid, _ in matched_by_view.get(2, []): right[gt].append(gid)
            for gt in sorted(set(left) & set(right)):
                if len(left[gt]) == 1 and len(right[gt]) == 1:
                    cross_total += 1
                    if left[gt][0] == right[gt][0]: cross_equal += 1
                    else: cross_conflicts.append({"frame": frame, "gt_instance_id": gt, "view1": left[gt][0], "view2": right[gt][0]})
        frame_records.append({"frame": frame, "matched": sum(len(v) for v in matched_by_view.values()), "cross_view_pairs": cross_total})
    return {
        "frames": len(rows), "gt_boxes": gt_total, "visible_predictions": pred_visible,
        "matched_boxes": matched, "box_recall": float(matched / max(gt_total, 1)),
        "box_precision": float(matched / max(pred_visible, 1)), "id_switches": switches,
        "fragments": fragments, "cross_view_pairs": cross_total,
        "cross_view_same_global_id": cross_equal,
        "cross_view_accuracy": float(cross_equal / max(cross_total, 1)),
        "cross_view_conflicts_sample": cross_conflicts[:50], "frame_records": frame_records,
        "iou_threshold": float(iou_threshold),
    }
