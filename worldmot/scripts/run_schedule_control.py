#!/usr/bin/env python
"""Finite execution-order control for the detector/Omega input pair.

Both schedules consume the same complete RGB pair and run the same pinned
models.  The control is diagnostic only; it does not enter the A/B0 ranking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worldmot.data.synchronized_loader import load_pair
from worldmot.data.visiontrack import VisionTrack
from worldmot.models.omega_geometry_backend import OmegaGeometryBackend
from worldmot.models.person_detector import PersonDetector


def _det_signature(dets):
    return [(int(v), [(round(float(x.bbox_xywh[0]), 4), round(float(x.bbox_xywh[1]), 4),
                       round(float(x.bbox_xywh[2]), 4), round(float(x.bbox_xywh[3]), 4),
                       round(float(x.score), 6)) for x in ds]) for v, ds in sorted(dets.items())]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="/data/DATASETS/TRACKING/JDE/VisionTrack")
    ap.add_argument("--scene", type=int, default=8); ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--omega-source", required=True)
    ap.add_argument("--omega-checkpoint", required=True); ap.add_argument("--det-config", required=True)
    ap.add_argument("--det-checkpoint", required=True)
    ap.add_argument("--output", default="results/detection_order/schedule_control.json")
    args = ap.parse_args()
    dataset = VisionTrack(args.dataset_root)
    detector = PersonDetector(args.det_config, args.det_checkpoint, device=args.device)
    omega = OmegaGeometryBackend(args.omega_source, args.omega_checkpoint, device=args.device)
    records = []
    for pair in dataset.synchronized_pairs(args.scene, start_frame=1, limit=args.limit):
        images = load_pair(dataset, pair)
        t0 = time.perf_counter()
        d_first = detector.infer(images)
        g_after = omega.reconstruct(images, pair.frame_id)
        detect_first_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        g_first = omega.reconstruct(images, pair.frame_id)
        d_after = detector.infer(images)
        omega_first_s = time.perf_counter() - t1
        depth_delta = 0.0
        for view in sorted(g_after.pointmaps):
            depth_delta = max(depth_delta, float(np.nanmax(np.abs(g_after.pointmaps[view] - g_first.pointmaps[view]))))
        records.append({
            "frame": pair.frame_id, "same_detector_outputs": _det_signature(d_first) == _det_signature(d_after),
            "same_geometry_shapes": {str(v): list(g_after.pointmaps[v].shape) == list(g_first.pointmaps[v].shape)
                                      for v in g_after.pointmaps},
            "max_pointmap_abs_delta": depth_delta,
            "detector_first_s": detect_first_s, "omega_first_s": omega_first_s,
        })
    result = {
        "state": "PASS" if all(x["same_detector_outputs"] and all(x["same_geometry_shapes"].values()) and
                                  x["max_pointmap_abs_delta"] < 1e-4 for x in records) else "MISMATCH",
        "scene": args.scene, "frames": len(records), "records": records,
        "interpretation": "execution order only; no ranking or GMT integration",
        "detector": detector.manifest(), "omega": omega.diagnostics(),
    }
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"state": result["state"], "frames": len(records), "output": str(path)}))
    return 0 if result["state"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
