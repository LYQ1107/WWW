#!/usr/bin/env python
"""Render per-query 3-D supports and 2-D masks for one real PanSt3R pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from worldmot.data.synchronized_loader import load_pair
from worldmot.data.visiontrack import VisionTrack
from worldmot.models.panoptic_instances import build_world_instances


PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 190), (0, 128, 128), (230, 190, 255),
]


def color_for(index: int) -> tuple[int, int, int]:
    return PALETTE[index % len(PALETTE)]


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    # PIL expects (width, height), while arrays are (height, width).
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST)) > 0


def draw_2d_overlay(image: np.ndarray, instances, view: int, output: Path) -> None:
    import cv2

    canvas = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    h, w = canvas.shape[:2]
    all_masks = []
    for index, instance in enumerate(instances):
        if view not in instance.masks_by_camera:
            continue
        mask_model = np.asarray(instance.masks_by_camera[view]).astype(bool)
        mask = resize_mask(mask_model, (w, h))
        if not mask.any():
            continue
        rgb = np.asarray(color_for(index), dtype=np.uint8)
        color_bgr = rgb[::-1]
        canvas[mask] = (0.48 * canvas[mask] + 0.52 * color_bgr).astype(np.uint8)
        ys, xs = np.where(mask)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        cv2.rectangle(canvas, (x0, y0), (x1, y1), tuple(int(x) for x in color_bgr), 3)
        label = f"q{instance.proposal_id}  {len(instance.xyz_support)}pts"
        cv2.putText(canvas, label, (x0, max(24, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, tuple(int(x) for x in color_bgr), 2, cv2.LINE_AA)
        all_masks.append({"proposal_id": int(instance.proposal_id), "color_rgb": list(map(int, rgb)),
                          "support_points": int(len(instance.xyz_support)), "visible": True})
    cv2.putText(canvas, f"view {view}: shared-query instances", (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, f"view {view}: shared-query instances", (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.imwrite(str(output), canvas)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panst3r-root", required=True)
    ap.add_argument("--must3r-root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset-root", default="/data/DATASETS/TRACKING/JDE/VisionTrack")
    ap.add_argument("--scene", type=int, default=8)
    ap.add_argument("--frame", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-dir", default="results/worldmot_smoke_v2")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset = VisionTrack(args.dataset_root)
    pair = next(dataset.synchronized_pairs(args.scene, start_frame=args.frame, limit=1))
    images = load_pair(dataset, pair)
    from worldmot.models.panst3r_backbone import PanSt3RBackend
    backend = PanSt3RBackend(args.panst3r_root, args.must3r_root, args.weights, device=args.device, image_size=512)
    frame = backend.reconstruct(images, pair.frame_id)
    instances = build_world_instances(frame.pointmaps, frame.confidences, frame.panoptic, frame.query_features)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # The gray points provide scene context; each colored cluster is exactly
    # the support sampled from one native shared-query mask.
    fig = plt.figure(figsize=(11, 8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    for view in sorted(frame.pointmaps):
        points = np.asarray(frame.pointmaps[view]).reshape(-1, 3)
        conf = np.asarray(frame.confidences[view]).reshape(-1)
        good = np.isfinite(points).all(1) & np.isfinite(conf) & (conf >= 1.5)
        points = points[good]
        if len(points) > 4500:
            points = points[np.linspace(0, len(points) - 1, 4500, dtype=np.int64)]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.18, alpha=0.10, c="#a9a9a9")
    metadata_instances = []
    for index, instance in enumerate(instances):
        pts = np.asarray(instance.xyz_support, dtype=np.float64)
        rgb = np.asarray(color_for(index), dtype=np.float64) / 255.0
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.8, alpha=0.82,
                   c=[rgb], label=f"q{instance.proposal_id} ({len(pts)} pts)")
        center = np.asarray(instance.xyz_center)
        ax.scatter([center[0]], [center[1]], [center[2]], marker="*", s=95,
                   c=[rgb], edgecolors="black")
        ax.text(center[0], center[1], center[2], f"q{instance.proposal_id}", fontsize=8)
        metadata_instances.append({
            "proposal_id": int(instance.proposal_id),
            "visible_cameras": list(map(int, instance.visible_cameras)),
            "support_points": int(len(pts)),
            "color_rgb": list(map(int, color_for(index))),
            "center": center.tolist(),
        })
    ax.set_title(f"Per-query 3-D instance supports — Scene {args.scene}, frame {args.frame}")
    ax.set_xlabel("local/world X")
    ax.set_ylabel("local/world Y")
    ax.set_zlabel("local/world Z")
    ax.view_init(elev=22, azim=-65)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    pointcloud_path = outdir / "pointcloud_instances_by_query.png"
    fig.savefig(pointcloud_path, bbox_inches="tight")
    plt.close(fig)

    overlay_meta = {}
    for view, image in images.items():
        path = outdir / f"view{view}_instances_by_query.png"
        draw_2d_overlay(image, instances, int(view), path)
        overlay_meta[str(view)] = str(path)
    metadata = {
        "scene": int(args.scene),
        "frame": int(args.frame),
        "status": frame.status,
        "source": "official PanSt3R v2 shared-query masks and pointmaps; no GT or GMT data",
        "pointcloud": str(pointcloud_path),
        "overlays": overlay_meta,
        "instances": metadata_instances,
        "interpretation": "Each colored 3-D cluster is the support sampled from one native shared query. Query ids are per-forward proposal ids, not persistent track ids.",
    }
    (outdir / "instance_visualization.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
