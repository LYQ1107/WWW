#!/usr/bin/env python
"""Render one saved detection-order frame (RGB boxes, 3-D instances, reprojection)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worldmot.data.visiontrack import VisionTrack


def _box(draw, box, color, label):
    x, y, w, h = [float(v) for v in box]
    draw.rectangle((x, y, x + w, y + h), outline=color, width=3)
    draw.text((x + 2, max(0, y - 15)), str(label), fill=color)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="/data/DATASETS/TRACKING/JDE/VisionTrack")
    ap.add_argument("--system", choices=("A", "B0"), default="A")
    ap.add_argument("--scene", type=int, default=8)
    ap.add_argument("--frame", type=int, default=1)
    ap.add_argument("--output", default="results/detection_order/visuals")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    jsonl = root / "results" / "detection_order" / args.system / f"scene_{args.scene:02d}" / "tracks.jsonl"
    rows = [json.loads(x) for x in jsonl.read_text(encoding="utf-8").splitlines() if x.strip()]
    row = next((x for x in rows if int(x["timestamp"]) == args.frame), rows[0])
    dataset = VisionTrack(args.dataset_root)
    out = root / args.output / f"{args.system.lower()}_scene_{args.scene:02d}_frame_{int(row['timestamp']):06d}"
    out.mkdir(parents=True, exist_ok=True)
    for view, image_id in row.get("image_ids", {}).items():
        rec = dataset.images[int(image_id)]
        image = Image.open(dataset.image_path(rec)).convert("RGB")
        draw = ImageDraw.Draw(image)
        for det in row.get("detections_by_camera", {}).get(str(view), []):
            _box(draw, det["bbox_xywh"], (255, 90, 0), f"det {det['detection_id']} {det['score']:.2f}")
        for tr in row.get("tracks_2d_by_camera", {}).get(str(view), []):
            if tr.get("bbox_xywh") is not None:
                color = (0, 220, 80) if tr.get("visible") else (100, 150, 255)
                _box(draw, tr["bbox_xywh"], color, f"ID {tr['global_id']}")
        # Original GT is shown only as an audit overlay in yellow; it is not
        # read by the causal runtime and is never used to construct output.
        for ann in dataset.ground_truth(int(image_id)):
            _box(draw, ann["bbox"], (255, 220, 0), f"GT {ann.get('instance_id', '?')}")
        image.save(out / f"view_{int(view)}.png")
    centers = []
    labels = []
    for obs in row.get("world_observations", []):
        c = np.asarray(obs.get("center", [np.nan] * 3), dtype=float)
        if np.isfinite(c).all():
            centers.append(c); labels.append(obs.get("observation_id", "obs"))
    if not centers:
        for tr in row.get("tracks_3d", []):
            c = np.asarray(tr.get("xyz_center", [np.nan] * 3), dtype=float)
            if np.isfinite(c).all():
                centers.append(c); labels.append(f"ID {tr.get('global_id')}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 6), dpi=160)
        ax = fig.add_subplot(111, projection="3d")
        if centers:
            c = np.asarray(centers)
            ax.scatter(c[:, 0], c[:, 1], c[:, 2], s=55, c=np.arange(len(c)), cmap="tab10")
            for p, label in zip(c, labels):
                ax.text(float(p[0]), float(p[1]), float(p[2]), str(label), fontsize=7)
        ax.set_xlabel("world X"); ax.set_ylabel("world Y"); ax.set_zlabel("world Z")
        ax.set_title(f"{args.system} Scene {args.scene} frame {row['timestamp']} 3-D instances")
        fig.tight_layout(); fig.savefig(out / "world_instances_3d.png"); plt.close(fig)
    except Exception as exc:
        (out / "world_instances_3d.txt").write_text(f"3-D renderer unavailable: {exc}\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "system": args.system, "scene": args.scene, "frame": row["timestamp"],
        "image_ids": row.get("image_ids"), "world_observation_count": len(row.get("world_observations", [])),
        "track_count": len(row.get("tracks_3d", [])),
        "note": "yellow GT is an audit overlay only; green/orange/blue are saved predictions and projections",
    }, indent=2), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
