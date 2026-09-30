#!/usr/bin/env python3
"""Build the descriptive baseline observation table for A1/A4/A5.

Matching is diagnostic Hungarian matching with IoU >= 0.5.  It is explicitly
not the official HOTA/IDF1/CVMA/CVIDF1 matching implementation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a.T
    bx1, by1, bx2, by2 = b.T
    xx1 = np.maximum(ax1[:, None], bx1[None, :])
    yy1 = np.maximum(ay1[:, None], by1[None, :])
    xx2 = np.minimum(ax2[:, None], bx2[None, :])
    yy2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    area_a = np.maximum(0, ax2 - ax1) * np.maximum(0, ay2 - ay1)
    area_b = np.maximum(0, by2 - by1) * np.maximum(0, by2 - by1)
    # Correct the second dimension explicitly; the expression above is kept
    # separate to make the box convention obvious.
    area_b = np.maximum(0, bx2 - bx1) * np.maximum(0, by2 - by1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def blur_for_crop(image: np.ndarray, box: tuple[float, float, float, float]) -> float:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    ix1, iy1 = max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))
    ix2, iy2 = min(w, int(math.ceil(x2))), min(h, int(math.ceil(y2)))
    if ix2 <= ix1 or iy2 <= iy1:
        return float("nan")
    crop = image[iy1:iy2, ix1:ix2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    longest = max(gray.shape[:2])
    if longest > 128:
        scale = 128.0 / longest
        gray = cv2.resize(gray, (max(3, int(gray.shape[1] * scale)), max(3, int(gray.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--border-tolerance-px", type=float, default=2.0)
    ap.add_argument("--scene", default=None, help="optional single scene shard")
    args = ap.parse_args()

    data = json.loads(args.annotations.read_text())
    videos = {x["id"]: x for x in data["videos"]}
    images = {x["id"]: x for x in data["images"]}
    anns_by_image: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    # Predictions are the 44 released MOT text files: scene/View1.txt etc.
    preds: dict[tuple[str, int, int], list[tuple[int, float, float, float, float, float]]] = {}
    for scene_dir in sorted(args.predictions.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene = scene_dir.name
        for txt in sorted(scene_dir.glob("View*.txt")):
            view = int(txt.stem.replace("View", ""))
            for line in txt.read_text().splitlines():
                if not line.strip():
                    continue
                fields = [float(x) for x in line.split(",")]
                if len(fields) < 7:
                    raise ValueError(f"malformed prediction row: {txt}: {line}")
                frame, track_id = int(fields[0]), int(fields[1])
                preds.setdefault((scene, view, frame), []).append(
                    (track_id, fields[2], fields[3], fields[4], fields[5], fields[6])
                )

    image_by_key = {}
    for image_id, image in images.items():
        scene = videos[image["video_id"]]["file_name"]
        image_by_key[(scene, int(image["view_id"]), int(image["frame_id"]))] = image

    rows = []
    matched = 0
    for key in sorted(preds):
        scene, view, frame = key
        if args.scene is not None and scene != args.scene:
            continue
        image = image_by_key.get(key)
        if image is None:
            continue
        gt = anns_by_image.get(image["id"], [])
        p = preds[key]
        gt_boxes = np.array(
            [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in gt],
            dtype=np.float32,
        ).reshape((-1, 4))
        pred_boxes = np.array([[x[1], x[2], x[3], x[4]] for x in p], dtype=np.float32).reshape((-1, 4))
        ious = iou_matrix(gt_boxes, pred_boxes)
        if len(gt_boxes) and len(pred_boxes):
            gi, pi = linear_sum_assignment(1.0 - ious)
        else:
            gi, pi = np.array([], dtype=int), np.array([], dtype=int)
        image_path = args.image_root / image["file_name"]
        cache_key = str(image_path)
        # Each key corresponds to one image.  Read it once for this frame and
        # release it at the end of the key; retaining all frames would consume
        # hundreds of GB of RAM.
        image_array = cv2.imread(cache_key, cv2.IMREAD_COLOR)
        for gidx, pidx in zip(gi, pi):
            ov = float(ious[gidx, pidx])
            if ov < args.iou_threshold:
                continue
            track_id, x1, y1, x2, y2, score = p[pidx]
            width, height = float(image["width"]), float(image["height"])
            bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
            blur = blur_for_crop(image_array, (x1, y1, x2, y2)) if image_array is not None else float("nan")
            tol = args.border_tolerance_px
            touch = int(x1 <= tol or y1 <= tol or x2 >= width - tol or y2 >= height - tol)
            rows.append({
                "scene": scene,
                "video_id": int(image["video_id"]),
                "image_id": int(image["id"]),
                "frame_id": int(frame),
                "view_id": int(view),
                "gt_instance_id": int(gt[gidx]["instance_id"]),
                "pred_track_id": int(track_id),
                "pred_score": float(score),
                "iou": ov,
                "bbox_x": float(x1),
                "bbox_y": float(y1),
                "bbox_w": bw,
                "bbox_h": bh,
                "image_w": int(width),
                "image_h": int(height),
                "normalized_area": bw * bh / max(width * height, 1.0),
                "sqrt_normalized_area": math.sqrt(bw * bh / max(width * height, 1.0)),
                "touch_border": touch,
                "crop_blur": blur,
            })
            matched += 1

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    manifest = {
        "annotations": str(args.annotations),
        "predictions": str(args.predictions),
        "image_root": str(args.image_root),
        "output": str(args.output),
        "matching": "Hungarian cost=1-IoU, accept IoU >= %.3f" % args.iou_threshold,
        "border_tolerance_px": args.border_tolerance_px,
        "diagnostic_not_official_matching": True,
        "matched_observations": matched,
        "rows": len(df),
        "scenes": sorted(df.scene.unique().tolist()) if len(df) else [],
        "scene_filter": args.scene,
        "columns": list(df.columns),
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
