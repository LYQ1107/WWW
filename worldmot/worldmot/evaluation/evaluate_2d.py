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
    matched = 0
    gt_total = 0
    pred_visible = 0
    switches = 0
    fragments = 0
    prev_by_gt: dict[tuple[int, int, int], int] = {}
    last_match_index: dict[tuple[int, int, int], int] = {}
    pair_events: list[dict] = []
    cross_total = 0
    cross_equal = 0
    cross_conflicts = []
    frame_records = []
    for row_index, row in enumerate(rows):
        scene = int(row.get("scene_id") or row.get("scene", 0))
        frame = int(row["timestamp"])
        pair_gt = {
            int(view): dataset.ground_truth(int(image_id))
            for view, image_id in row.get("image_ids", {}).items()
        }
        matched_by_view = {}
        for view, anns in pair_gt.items():
            gt_total += len(anns)
            preds = row.get("tracks_2d_by_camera", {}).get(str(view), row.get("tracks_2d_by_camera", {}).get(view, []))
            # ``visible`` is deliberately evaluation-eligible only for a
            # current target-view observation; projection-only diagnostics are
            # excluded from FP/TP counts.
            eligible = [p for p in preds if p.get("visible") and p.get("bbox_xywh") is not None and
                        p.get("evaluation_eligible", True)]
            pred_visible += len(eligible)
            pairs = _match(eligible, anns, iou_threshold)
            matched += len(pairs)
            matched_by_view[view] = []
            for pi, ai, score in pairs:
                gt_id = int(anns[ai].get("instance_id", anns[ai].get("track_id", -1)))
                gid = int(eligible[pi]["global_id"])
                matched_by_view[view].append((gt_id, gid, float(score)))
                key = (scene, int(view), gt_id)
                if key in prev_by_gt and prev_by_gt[key] != gid:
                    switches += 1
                if key in last_match_index and last_match_index[key] != row_index - 1:
                    fragments += 1
                prev_by_gt[key] = gid
                last_match_index[key] = row_index
                pair_events.append({"row": row_index, "scene": scene, "frame": frame, "view": int(view),
                                    "gt": gt_id, "pred": gid, "iou": float(score)})
        if len(matched_by_view) >= 2:
            left = defaultdict(list); right = defaultdict(list)
            for gt, gid, _ in matched_by_view.get(1, []): left[gt].append(gid)
            for gt, gid, _ in matched_by_view.get(2, []): right[gt].append(gid)
            for gt in sorted(set(left) & set(right)):
                if len(left[gt]) == 1 and len(right[gt]) == 1:
                    cross_total += 1
                    if left[gt][0] == right[gt][0]:
                        cross_equal += 1
                    else:
                        cross_conflicts.append({"frame": frame, "gt_instance_id": gt,
                                                "view1": left[gt][0], "view2": right[gt][0]})
        frame_records.append({"frame": frame, "matched": sum(len(v) for v in matched_by_view.values()),
                              "pred_visible": sum(len([p for p in row.get("tracks_2d_by_camera", {}).get(str(v), [])
                                                        if p.get("visible") and p.get("evaluation_eligible", True)])
                                                  for v in pair_gt),
                              "cross_view_pairs": cross_total})

    # Identity metrics use all frames of this sequence, with the same fixed
    # IoU assignment as the detection statistics.  They are the standard
    # IDF1/AssA/HOTA definitions evaluated on the supplied complete rows.
    gt_counts = defaultdict(int); pred_counts = defaultdict(int); pair_counts = defaultdict(int)
    for event in pair_events:
        key_gt = (event["scene"], event["view"], event["gt"])
        key_pred = (event["scene"], event["view"], event["pred"])
        gt_counts[key_gt] += 1; pred_counts[key_pred] += 1
        pair_counts[(key_gt, key_pred)] += 1
    assigned = 0
    if pair_counts:
        from scipy.optimize import linear_sum_assignment
        gt_keys = sorted(gt_counts); pred_keys = sorted(pred_counts)
        mat = np.zeros((len(gt_keys), len(pred_keys)), dtype=np.float64)
        for (g, p), count in pair_counts.items():
            mat[gt_keys.index(g), pred_keys.index(p)] = count
        ri, ci = linear_sum_assignment(-mat)
        assigned = int(mat[ri, ci].sum())
    idfp = max(0, pred_visible - assigned); idfn = max(0, gt_total - assigned)
    idf1 = float(2 * assigned / max(2 * assigned + idfp + idfn, 1))
    ass_values = []
    for (g, p), tp in pair_counts.items():
        fn = gt_counts[g] - tp; fp = pred_counts[p] - tp
        ass_values.extend([float(tp / max(tp + fn + fp, 1))] * int(tp))
    assa = float(np.mean(ass_values)) if ass_values else 0.0
    deta = float(matched / max(gt_total + pred_visible - matched, 1))
    hota = float(np.sqrt(max(0.0, deta * assa)))
    cross_fp = max(0, cross_total - cross_equal)
    cvidf1 = float(2 * cross_equal / max(2 * cross_equal + cross_fp, 1))
    return {
        "frames": len(rows), "gt_boxes": gt_total, "visible_predictions": pred_visible,
        "matched_boxes": matched, "false_positives": max(0, pred_visible - matched),
        "false_negatives": max(0, gt_total - matched),
        "box_recall": float(matched / max(gt_total, 1)),
        "box_precision": float(matched / max(pred_visible, 1)),
        "visibility_accuracy": float(matched / max(gt_total, 1)),
        "id_switches": switches, "fragments": fragments,
        "IDF1": idf1, "AssA": assa, "HOTA": hota, "DetA": deta,
        "CVIDF1": cvidf1, "cross_view_pairs": cross_total,
        "cross_view_same_global_id": cross_equal,
        "cross_view_accuracy": float(cross_equal / max(cross_total, 1)),
        "cross_view_conflicts_sample": cross_conflicts[:50], "frame_records": frame_records,
        "matched_iou_mean": float(np.mean([x["iou"] for x in pair_events])) if pair_events else 0.0,
        "iou_threshold": float(iou_threshold),
        "AP": None,
        "AP_reason": "single operating-point JSONL has no score sweep; use detector-native COCO AP separately",
        "protocol": "complete supplied sequence, original VisionTrack GT, fixed IoU, one-to-one greedy frame assignment",
    }
