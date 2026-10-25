#!/usr/bin/env python3
"""Live RGB -> Ω -> persistent world-object replay for E2E timing.

This runner deliberately does not read the WorldObject support cache.  It
reuses only the immutable GMT candidate cache and the original RGB frames, so
its timing and output can be compared against the cache-replay route on the
same causal frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from worldmot.data.visiontrack import VisionTrack
from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
from worldmot.models.omega_geometry_backend import OmegaGeometryBackend
from worldmot.runtime.online_world_object_tracker import OnlineWorldObjectTracker


DEFAULT_OMEGA = "/data1/liuyeqiang/WWW_omega_runtime_20261016/official_vggt_omega"
DEFAULT_WEIGHTS = "/data1/liuyeqiang/WWW_omega_runtime_20261016/weights/vggt_omega_1b_512.pt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_safe(value):
    """Convert diagnostic NaN/Inf values to JSON null without hiding them.

    Ω confidence and alignment residuals can be non-finite for an abstained
    frame.  The online result must still be persisted so that the timing and
    failure reason remain auditable; strict JSON rejects those IEEE values.
    """
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def claim(queue: Path) -> int | None:
    for pending in sorted(queue.glob("scene_*.pending")):
        running = pending.with_suffix(".running")
        try:
            os.replace(pending, running)
        except FileNotFoundError:
            continue
        return int(pending.stem.split("_")[1])
    return None


def read_images(ds, pair):
    return {int(v): np.asarray(Image.open(ds.image_path(rec)).convert("RGB"))
            for v, rec in pair.images.items()}


def run_scene(ds, sid, detector, geometry, out_scene, max_timestamps=None):
    out_scene.mkdir(parents=True, exist_ok=True)
    tracker = OnlineWorldObjectTracker(detector, geometry, device=geometry.device_name)
    tracker.reset(int(sid)); tracks = []; timings = []
    for pair_index, pair in enumerate(ds.synchronized_pairs(int(sid))):
        if max_timestamps is not None and pair_index >= int(max_timestamps):
            break
        e2e_started = time.perf_counter()
        rgb_started = time.perf_counter()
        images = read_images(ds, pair)
        rgb_read_s = float(time.perf_counter() - rgb_started)
        detector.set_context(int(sid), int(pair.frame_id),
                             {int(v): int(rec.image_id) for v, rec in pair.images.items()})
        row = tracker.step(images, int(pair.frame_id))
        # Include synchronous RGB decode/read and all tracker stages.  This is
        # the online wall-clock definition; it must not be confused with the
        # cache-replay or pure Ω-kernel timings.
        wall = float(time.perf_counter() - e2e_started)
        tracks.append(json_safe(row))
        timings.append(json_safe({"timestamp": int(pair.frame_id), "live_e2e_s": wall,
                        "rgb_read_s": rgb_read_s,
                        "stages": row.get("runtime", {}).get("stages", {})}))
    with (out_scene / "tracks.jsonl").open("w") as f:
        for row in tracks:
            f.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
    with (out_scene / "timings.jsonl").open("w") as f:
        for row in timings:
            f.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
    manifest = {
        "schema": "worldobject_live_prediction_v1", "route": "LIVE_RGB_TO_ID",
        "scene": int(sid), "status": "COMPLETE", "timestamps": len(tracks),
        "first_frame": tracks[0]["timestamp"] if tracks else None,
        "last_frame": tracks[-1]["timestamp"] if tracks else None,
        "causal": True, "GT_in_inference": False,
        "tracks_sha256": sha256(out_scene / "tracks.jsonl"),
        "timings_sha256": sha256(out_scene / "timings.jsonl"),
        "fps_definition": "live_rgb_to_global_id",
        "sample_limit": int(max_timestamps) if max_timestamps is not None else None,
        "detector": detector.manifest(), "omega": geometry.diagnostics(),
    }
    (out_scene / "manifest.json").write_text(json.dumps(json_safe(manifest), indent=2, allow_nan=False))
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="/data/DATASETS/TRACKING/JDE/VisionTrack")
    p.add_argument("--raw-cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--scenes", nargs="*", type=int)
    p.add_argument("--scene-queue")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-timestamps", type=int,
                   help="optional registered live timing sample limit; does not produce a full-video accuracy result")
    p.add_argument("--omega-source", default=DEFAULT_OMEGA)
    p.add_argument("--omega-weights", default=DEFAULT_WEIGHTS)
    a = p.parse_args()
    ds = VisionTrack(a.dataset_root, split=a.split, load_annotations=False)
    detector = ImmutableGMTDetectionCache(a.raw_cache)
    geometry = OmegaGeometryBackend(a.omega_source, a.omega_weights, device=a.device,
                                    expose_intermediate_features=False)
    if geometry.status != "READY":
        raise RuntimeError(f"Omega unavailable: {geometry.reason}")
    root = Path(a.output); root.mkdir(parents=True, exist_ok=True)
    reports = []
    if a.scene_queue:
        queue = Path(a.scene_queue)
        while True:
            sid = claim(queue)
            if sid is None: break
            try:
                reports.append(run_scene(ds, sid, detector, geometry, root / f"scene_{sid:02d}", a.max_timestamps))
                (queue / f"scene_{sid:02d}.running").unlink(missing_ok=True)
                (queue / f"scene_{sid:02d}.done").touch()
            except Exception:
                (queue / f"scene_{sid:02d}.failed").write_text("failed")
                raise
    else:
        for sid in a.scenes or ds.scene_ids():
            reports.append(run_scene(ds, int(sid), detector, geometry, root / f"scene_{int(sid):02d}", a.max_timestamps))
    (root / "coverage.json").write_text(json.dumps(json_safe({
        "schema": "worldobject_live_coverage_v1", "route": "LIVE_RGB_TO_ID",
        "split": a.split, "scenes": reports, "scene_count": len(reports),
        "all_complete": all(x["status"] == "COMPLETE" for x in reports),
    }), indent=2, allow_nan=False))
    print(json.dumps({"scenes": len(reports), "complete": all(x["status"] == "COMPLETE" for x in reports)}, indent=2))


if __name__ == "__main__":
    main()

