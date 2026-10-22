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


def _compact_record(row: dict) -> dict:
    """Keep only fields required for reproducible metrics and failure audits.

    The normal runner preserves full manifests for single-scene forensic work.
    Full-scene sweeps use this projection so repeated detector/Omega manifests
    do not turn every synchronized timestamp into a multi-hundred-kilobyte
    JSON record.
    """
    detections = {}
    for view, values in row.get("detections_by_camera", {}).items():
        detections[str(view)] = [
            {"bbox_xywh": list(d.get("bbox_xywh", [])), "score": float(d.get("score", 0.0))}
            for d in values
        ]
    observations = []
    for obs in row.get("world_observations", []):
        observations.append({
            "fusion_status": obs.get("fusion_status", "UNKNOWN"),
            "geometry_reliable": bool(obs.get("geometry_reliable", False)),
        })
    tracks = {}
    for view, values in row.get("tracks_2d_by_camera", {}).items():
        tracks[str(view)] = [
            {
                "global_id": int(t["global_id"]),
                "visible": bool(t.get("visible", False)),
                "evaluation_eligible": bool(t.get("evaluation_eligible", False)),
                "bbox_xywh": list(t["bbox_xywh"]) if t.get("bbox_xywh") is not None else None,
            }
            for t in values
        ]
    runtime = row.get("runtime", {}) or {}
    return {
        "timestamp": int(row["timestamp"]),
        "scene_id": row.get("scene_id"),
        "detections_by_camera": detections,
        "world_observations": observations,
        "tracks_2d_by_camera": tracks,
        "geometry_status": row.get("geometry_status"),
        "alignment": row.get("alignment", {}),
        "fusion": row.get("fusion", {}),
        "runtime": {
            "frame_stages": runtime.get("frame_stages", {}),
            "frame_total_s": float(runtime.get("frame_total_s", 0.0)),
        },
        "proposal_count": int(row.get("proposal_count", 0)),
        "causal": bool(row.get("causal", True)),
        "ablation": row.get("ablation"),
        "image_ids": row.get("image_ids", {}),
        "rgb_read_s": float(row.get("rgb_read_s", 0.0)),
        "end_to_end_s": float(row.get("end_to_end_s", 0.0)),
    }


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
    ap.add_argument("--compact-output", action="store_true",
                    help="write metric/failure fields without repeated backend manifests")
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
                if args.compact_output:
                    row = _compact_record(row)
                f.write(json.dumps(row, ensure_ascii=False, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)) + "\n")
                f.flush()
                summary["records"].append({"frame": pair.frame_id, "geometry_status": row["geometry_status"],
                                           "alignment": row["alignment"], "proposal_count": row["proposal_count"],
                                           "track_count": len(row.get("tracks_3d", []))})
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

