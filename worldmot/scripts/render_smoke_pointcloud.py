#!/usr/bin/env python
"""Render a reviewable 3-D pointmap preview from one real PanSt3R pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from worldmot.data.synchronized_loader import load_pair
from worldmot.data.visiontrack import VisionTrack
from worldmot.models.panoptic_instances import build_world_instances


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

    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(11, 8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    colors = {1: "#1677ff", 2: "#f08a24"}
    point_count = {}
    for view in sorted(frame.pointmaps):
        points = np.asarray(frame.pointmaps[view]).reshape(-1, 3)
        confidence = np.asarray(frame.confidences[view]).reshape(-1)
        good = np.isfinite(points).all(axis=1) & np.isfinite(confidence) & (confidence >= 1.5)
        points = points[good]
        if len(points) > 6000:
            points = points[np.linspace(0, len(points) - 1, 6000, dtype=np.int64)]
        point_count[str(view)] = int(len(points))
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.25, alpha=0.22,
                   c=colors.get(int(view), "#666666"), label=f"camera {view} pointmap")
        pose = np.asarray(frame.cameras[view].world_from_camera)
        center = pose[:3, 3]
        ax.scatter([center[0]], [center[1]], [center[2]], marker="^", s=80,
                   c=colors.get(int(view), "#666666"), edgecolors="black")
    if instances:
        centers = np.asarray([x.xyz_center for x in instances])
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker="*", s=80,
                   c="#e31a1c", edgecolors="black", label="WorldInstance centers")
        for inst in instances:
            c = np.asarray(inst.xyz_center)
            ax.text(c[0], c[1], c[2], f"q{inst.proposal_id}", fontsize=7)
    ax.set_title(f"PanSt3R v2 pointmaps — VisionTrack Scene {args.scene}, frame {args.frame}")
    ax.set_xlabel("world/local X")
    ax.set_ylabel("world/local Y")
    ax.set_zlabel("world/local Z")
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=22, azim=-65)
    fig.tight_layout()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    image_path = outdir / "pointcloud_3d.png"
    fig.savefig(image_path, bbox_inches="tight")
    plt.close(fig)
    metadata = {
        "scene": args.scene,
        "frame": args.frame,
        "source": "official PanSt3R v2 pointmaps; no GT or GMT data",
        "point_count_after_filter": point_count,
        "world_instance_count": len(instances),
        "status": frame.status,
        "diagnostics": frame.diagnostics,
        "image": str(image_path),
    }
    (outdir / "pointcloud_3d.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
