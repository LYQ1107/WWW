#!/usr/bin/env python3
"""TRAIN-only evaluation of box vs foreground-aware world support."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from worldmot.data.visiontrack import VisionTrack
from worldmot.data.worldobject_cache import iter_complete_scenes
from worldmot.models.world_object_support import WorldObjectSupport, support_compatibility


def iou(a, b):
    ax, ay, aw, ah = map(float, a); bx, by, bw, bh = map(float, b)
    x0, y0 = max(ax, bx), max(ay, by); x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0., x1-x0) * max(0., y1-y0); union = max(0., aw)*max(0., ah)+max(0., bw)*max(0., bh)-inter
    return inter / union if union > 0 else 0.0


def match(rows, gt):
    pairs = sorted([(iou(r["bbox"], a.get("bbox", [0, 0, 0, 0])), i, j)
                    for i, r in enumerate(rows) for j, a in enumerate(gt)], reverse=True)
    used_i = set(); used_j = set(); out = {}
    for score, i, j in pairs:
        if score < .5 or i in used_i or j in used_j: continue
        used_i.add(i); used_j.add(j); out[i] = (gt[j], score)
    return out


def support(row):
    fg = np.asarray(row["foreground"], float); fg = fg[np.isfinite(fg).all(1)]
    box = np.asarray(row["box"], float); box = box[np.isfinite(box).all(1)]
    return WorldObjectSupport(box, fg, np.asarray(row["center"], float), np.asarray(row["cov"], float),
                              float(row["quality"]), float(row["foreground_fraction"]),
                              "CACHE", 0., 1.)


def size_group(bbox, width, height):
    area = max(0., float(bbox[2])) * max(0., float(bbox[3])) / max(1., width * height)
    return "small" if area < .02 else "medium" if area < .10 else "large"


def main():
    p = argparse.ArgumentParser(); p.add_argument("--cache", required=True); p.add_argument("--dataset-root", default="/data/DATASETS/TRACKING/JDE/VisionTrack"); p.add_argument("--output-dir", required=True)
    a = p.parse_args(); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True); ds = VisionTrack(a.dataset_root, split="train")
    object_rows=[]; breakdown=defaultdict(lambda: Counter()); all_summary=Counter(); scene_summary={}
    for cache in iter_complete_scenes(a.cache):
        frame_map = {int(p.frame_id): p for p in ds.synchronized_pairs(cache.scene)}; scene=Counter()
        for fr in cache.frames():
            pair = frame_map[fr["timestamp"]]; gt = {int(v): ds.ground_truth(rec.image_id) for v, rec in pair.images.items()}
            matched = {v: match([r for r in fr["rows"] if r["view"] == v], gt[v]) for v in gt}
            by_identity=defaultdict(list)
            for view, mm in matched.items():
                view_rows=[r for r in fr["rows"] if r["view"] == view]
                for idx, (ann, score) in mm.items(): by_identity[int(ann.get("instance_id", ann.get("track_id", ann.get("id", -1))))].append((view, view_rows[idx], score))
            for row in fr["rows"]:
                view_rows=[r for r in fr["rows"] if r["view"] == row["view"]]
                idx = next((i for i, x in enumerate(view_rows) if x["detection_id"] == row["detection_id"]), -1)
                m = matched.get(row["view"], {}).get(idx)
                key = "matched" if m else "unmatched"
                group = size_group(row["bbox"], pair.images[row["view"]].width, pair.images[row["view"]].height)
                q = {"scene": cache.scene, "timestamp": fr["timestamp"], "view": row["view"], "detection_id": row["detection_id"],
                     "matched_gt": bool(m), "matched_iou": float(m[1]) if m else 0., "size_group": group,
                     "quality": row["quality"], "foreground_fraction": row["foreground_fraction"],
                     "foreground_point_count": len(row["foreground"]), "box_point_count": len(row["box"]),
                     "ambiguity": {0: "NONE", 1: "DEPTH_CLUSTERED", 2: "DEPTH_NOT_SEPARATED", 3: "BOX_ONLY", 4: "LOW_SUPPORT", 5: "NO_FINITE_SUPPORT"}.get(row.get("ambiguity_code", -1), "UNKNOWN_CACHE")}
                object_rows.append(q); breakdown[group][key] += 1; breakdown[group]["low_quality" if row["quality"] < .3 else "high_quality"] += 1
                all_summary[key] += 1; scene[key] += 1
            # Same-person cross-view support consistency.
            for identity, obs in by_identity.items():
                if len(obs) < 2: continue
                left, right = obs[0][1], obs[1][1]; sl, sr = support(left), support(right)
                score, detail, reason = support_compatibility(sl, sr)
                scene["same_person_pairs"] += 1; scene["same_person_consistent"] += int(np.isfinite(score));
                all_summary["same_person_pairs"] += 1; all_summary["same_person_consistent"] += int(np.isfinite(score))
            # Different-person separability at the same timestamp.
            views = sorted({r["view"] for r in fr["rows"]})
            if len(views) >= 2:
                la=[r for r in fr["rows"] if r["view"] == views[0]]; rb=[r for r in fr["rows"] if r["view"] == views[1]]
                for l in la:
                    for r in rb:
                        ml = next((x for x in matched[views[0]].values() if iou(l["bbox"], x[0].get("bbox", [0]*4)) >= .5), None)
                        mr = next((x for x in matched[views[1]].values() if iou(r["bbox"], x[0].get("bbox", [0]*4)) >= .5), None)
                        if ml and mr and ml[0].get("instance_id") == mr[0].get("instance_id"): continue
                        score, _, _ = support_compatibility(support(l), support(r), max_normalized_distance=2.0)
                        scene["different_person_pairs"] += 1; all_summary["different_person_pairs"] += 1
                        if np.isfinite(score): scene["different_person_ambiguous"] += 1; all_summary["different_person_ambiguous"] += 1
        scene_summary[str(cache.scene)] = dict(scene)
    (out / "G1_object_support_all24.csv").write_text("")
    with (out / "G1_object_support_all24.csv").open("w", newline="") as f:
        fields=list(object_rows[0]) if object_rows else ["scene"]; w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(object_rows)
    rows=[]
    for group, c in sorted(breakdown.items()):
        rows.append({"size_group": group, "observations": sum(c.values()), "matched": c["matched"],
                     "unmatched": c["unmatched"], "low_quality": c["low_quality"], "high_quality": c["high_quality"]})
    with (out / "G1_object_support_size_crowd_breakdown.csv").open("w", newline="") as f:
        fields=list(rows[0]) if rows else ["size_group"]; w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    (out / "G1_object_support_summary.json").write_text(json.dumps({"schema":"worldobject_g1_support_v1", "scenes":scene_summary, "summary":dict(all_summary), "cache":str(Path(a.cache).resolve()), "gt_role":"ORACLE_GT_MATCH_DIAGNOSTIC_ONLY"}, indent=2))
    print(json.dumps({"scenes":len(scene_summary), "observations":len(object_rows), "same_person_pairs":all_summary["same_person_pairs"], "different_person_ambiguous":all_summary["different_person_ambiguous"]}, indent=2))


if __name__ == "__main__": main()
