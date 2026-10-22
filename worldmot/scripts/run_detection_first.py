#!/usr/bin/env python
"""Run the independent A (detect-first) WorldMOT ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worldmot.data.synchronized_loader import load_pair
from worldmot.data.visiontrack import VisionTrack
from worldmot.models.omega_geometry_backend import OmegaGeometryBackend
from worldmot.models.person_detector import PersonDetector
from worldmot.runtime.detection_first import DetectionFirstWorldMOT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="/data/DATASETS/TRACKING/JDE/VisionTrack")
    ap.add_argument("--scene", type=int, default=8)
    ap.add_argument("--start-frame", type=int, default=1)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--omega-source", required=True)
    ap.add_argument("--omega-checkpoint", required=True)
    ap.add_argument("--det-config", required=True)
    ap.add_argument("--det-checkpoint", required=True)
    ap.add_argument("--det-score", type=float, default=0.25)
    ap.add_argument("--image-resolution", type=int, default=512)
    ap.add_argument("--output-dir", default="results/detection_order/A")
    args = ap.parse_args()
    outdir = Path(args.output_dir) / f"scene_{args.scene:02d}"
    outdir.mkdir(parents=True, exist_ok=True)
    dataset = VisionTrack(args.dataset_root)
    summary = {"status": "STARTED", "system": "A_DETECT_FIRST_RTMDET_INS_TINY_OMEGA512",
               "scene": args.scene, "start_frame": args.start_frame, "limit": args.limit,
               "device": args.device, "records": []}
    jsonl = outdir / "tracks.jsonl"
    try:
        detector = PersonDetector(args.det_config, args.det_checkpoint, device=args.device,
                                  score_threshold=args.det_score)
        geometry = OmegaGeometryBackend(args.omega_source, args.omega_checkpoint,
                                         device=args.device, image_resolution=args.image_resolution)
        summary["detector"] = detector.manifest()
        summary["omega"] = geometry.diagnostics()
        if detector.status != "READY":
            raise RuntimeError(f"RTMDet unavailable: {detector.status}: {detector.reason}")
        if geometry.status != "READY":
            raise RuntimeError(f"Omega unavailable: {geometry.status}: {geometry.reason}")
        runtime = DetectionFirstWorldMOT(detector, geometry, device=args.device)
        runtime.reset(args.scene)
        with jsonl.open("w", encoding="utf-8") as f:
            for pair in dataset.synchronized_pairs(args.scene, start_frame=args.start_frame, limit=args.limit):
                t_read = time.perf_counter()
                images = load_pair(dataset, pair)
                read_s = time.perf_counter() - t_read
                row = runtime.step(images, pair.frame_id)
                row["image_ids"] = {str(v): int(x.image_id) for v, x in pair.images.items()}
                row["rgb_read_s"] = float(read_s)
                row["end_to_end_s"] = float(read_s + row["runtime"].get("frame_total_s", 0.0))
                row["runtime"].setdefault("frame_stages", {})["rgb_read"] = float(read_s)
                row["runtime"]["frame_stages"]["end_to_end"] = float(row["end_to_end_s"])
                f.write(json.dumps(row, ensure_ascii=False, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)) + "\n")
                f.flush()
                summary["records"].append({"frame": pair.frame_id, "geometry_status": row["geometry_status"],
                                           "alignment": row["alignment"], "proposal_count": row["proposal_count"],
                                           "track_count": len(row["tracks_3d"])})
        summary["status"] = "PASS" if len(summary["records"]) == args.limit else "PARTIAL"
        summary["runtime"] = runtime.timer.summary()
        summary["jsonl"] = str(jsonl)
    except Exception as exc:
        summary.update({"status": "PIPELINE_FAILED", "error": str(exc),
                        "traceback": traceback.format_exc(), "jsonl": str(jsonl)})
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] in ("PASS", "PARTIAL") else 2


if __name__ == "__main__":
    raise SystemExit(main())

