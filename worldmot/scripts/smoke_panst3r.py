#!/usr/bin/env python
"""Run and audit one real VisionTrack PanSt3R synchronized pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback
import time

import numpy as np

from worldmot.data.visiontrack import VisionTrack
from worldmot.data.synchronized_loader import load_pair
from worldmot.models.panoptic_instances import build_world_instances


def save_mask(mask, path: Path):
    from PIL import Image
    a = np.asarray(mask)
    if a.ndim > 2: a = a[0]
    Image.fromarray((a > 0).astype(np.uint8) * 255).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panst3r-root", required=True)
    ap.add_argument("--must3r-root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset-root", default="/data/DATASETS/TRACKING/JDE/VisionTrack")
    ap.add_argument("--scene", type=int, default=8)
    ap.add_argument("--frame", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--output", default="results/worldmot_smoke.json")
    args = ap.parse_args()
    outdir = Path(args.output).with_suffix("")
    outdir.mkdir(parents=True, exist_ok=True)
    result = {"scene": args.scene, "frame": args.frame, "status": None,
              "panst3r_root": str(Path(args.panst3r_root).resolve()), "weights": str(Path(args.weights).resolve())}
    try:
        dataset = VisionTrack(args.dataset_root)
        pair = next(dataset.synchronized_pairs(args.scene, start_frame=args.frame, limit=1))
        images = load_pair(dataset, pair)
        from worldmot.models.panst3r_backbone import PanSt3RBackend
        t0 = time.perf_counter()
        backend = PanSt3RBackend(args.panst3r_root, args.must3r_root, args.weights, device=args.device, image_size=args.image_size)
        frame = backend.reconstruct(images, pair.frame_id)
        instances = build_world_instances(frame.pointmaps, frame.confidences, frame.panoptic, frame.query_features)
        for view, image in images.items():
            from PIL import Image
            Image.fromarray(image).resize((512, 288)).save(outdir / f"view{view}_rgb.jpg", quality=90)
            pred = frame.panoptic[view]
            save_mask(pred["pan"], outdir / f"view{view}_panoptic.png")
        finite = {str(v): bool(np.isfinite(x).all()) for v, x in frame.pointmaps.items()}
        result.update({
            "status": frame.status if frame.status != "OK" or not instances else "PASS",
            "elapsed_s": time.perf_counter() - t0,
            "diagnostics": frame.diagnostics,
            "views": {str(v): {"image_shape": list(images[v].shape), "pointmap_shape": list(frame.pointmaps[v].shape),
                                "confidence_shape": list(frame.confidences[v].shape),
                                "segments": len(frame.panoptic[v]["segments_info"]), "finite": finite[str(v)]}
                       for v in images},
            "query_shape": list(frame.query_features.shape),
            "world_instances": [{"proposal_id": i.proposal_id, "center": i.xyz_center.tolist(),
                                 "support_count": len(i.xyz_support), "visible_cameras": list(i.visible_cameras),
                                 "confidence": i.confidence, "geometry_valid": i.geometry_valid}
                                for i in instances],
            "camera_models": {str(v): frame.cameras[v].to_dict() for v in frame.cameras},
        })
    except Exception as exc:
        result.update({"status": "DEPENDENCY_BLOCKED" if "import" in str(exc).lower() or "load" in str(exc).lower() else "SMOKE_FAILED",
                       "error": str(exc), "traceback": traceback.format_exc()})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] in ("PASS", "NO_PERSON_DETECTIONS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
