#!/usr/bin/env python
"""Run a causal continuous VisionTrack sequence and write auditable outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worldmot.data.visiontrack import VisionTrack
from worldmot.data.synchronized_loader import load_pair
from worldmot.models.panst3r_backbone import PanSt3RBackend
from worldmot.runtime.online_worldmot import OnlineWorldMOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panst3r-root", required=True); ap.add_argument("--must3r-root", required=True); ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset-root", default="/data/DATASETS/TRACKING/JDE/VisionTrack")
    ap.add_argument("--scene", type=int, default=8); ap.add_argument("--start-frame", type=int, default=1); ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--image-size", type=int, default=512); ap.add_argument("--output-dir", default="results/continuous")
    args = ap.parse_args()
    outdir = Path(args.output_dir) / f"scene_{args.scene:02d}"
    outdir.mkdir(parents=True, exist_ok=True)
    dataset = VisionTrack(args.dataset_root)
    result = {"status": "STARTED", "scene": args.scene, "start_frame": args.start_frame, "limit": args.limit, "records": []}
    try:
        pairs = dataset.synchronized_pairs(args.scene, start_frame=args.start_frame, limit=args.limit)
        backend = PanSt3RBackend(args.panst3r_root, args.must3r_root, args.weights, device=args.device, image_size=args.image_size)
        runtime = OnlineWorldMOT(backend, device=args.device)
        runtime.reset(args.scene)
        with (outdir / "tracks.jsonl").open("w", encoding="utf-8") as f:
            for pair in pairs:
                t_read = time.perf_counter()
                images = load_pair(dataset, pair)
                read_s = time.perf_counter() - t_read
                row = runtime.step(images, pair.frame_id)
                row["image_ids"] = {str(v): int(x.image_id) for v, x in pair.images.items()}
                row["rgb_read_s"] = float(read_s)
                row["end_to_end_s"] = float(read_s + row["runtime"].get("frame_total_s", 0.0))
                row["runtime"].setdefault("frame_stages", {})["rgb_read"] = float(read_s)
                row["runtime"]["frame_stages"]["end_to_end"] = float(row["end_to_end_s"])
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
                result["records"].append({"frame": pair.frame_id, "geometry_status": row["geometry_status"],
                                          "alignment": row["alignment"], "proposal_count": row["proposal_count"],
                                          "track_count": len(row["tracks_3d"])})
        result["status"] = "PASS" if len(result["records"]) == args.limit else "PARTIAL"
        result["runtime"] = runtime.timer.summary()
    except Exception as exc:
        result.update({"status": "PIPELINE_FAILED", "error": str(exc), "traceback": traceback.format_exc()})
    (outdir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] in ("PASS", "PARTIAL") else 2


if __name__ == "__main__":
    raise SystemExit(main())
