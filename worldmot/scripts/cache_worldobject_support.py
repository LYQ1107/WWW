#!/usr/bin/env python3
"""Build a resumable sparse Ω support cache for the WorldObject experiment."""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from worldmot.data.visiontrack import VisionTrack
from worldmot.geometry.dual_view_static_alignment import DualCameraStaticAligner
from worldmot.geometry.world_alignment import Sim3Aligner
from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
from worldmot.models.omega_geometry_backend import OmegaGeometryBackend
from worldmot.runtime.detection_first import DetectionFirstWorldMOT
from worldmot.runtime.online_worldmot import OnlineWorldMOT
from worldmot.runtime.experiment_authorization import require_experiment_authorization, source_commit,source_fingerprint
from worldmot.geometry.world_frame_contract import WorldFrameTransform
from worldmot.data.worldobject_observations import live_observations


SCHEMA = "worldobject_omega512_sparse_support_v2"
DEFAULT_OMEGA = "/data1/liuyeqiang/WWW_omega_runtime_20261016/official_vggt_omega"
DEFAULT_OMEGA_WEIGHTS = "/data1/liuyeqiang/WWW_omega_runtime_20261016/weights/vggt_omega_1b_512.pt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path = Path(path); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)); os.replace(tmp, path)


def safe_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def read_images(ds, pair):
    return {int(v): np.asarray(Image.open(ds.image_path(rec)).convert("RGB"))
            for v, rec in pair.images.items()}


class SceneState:
    def __init__(self, omega, detector_cache, scene_id):
        self.omega = omega; self.detector_cache = detector_cache; self.scene_id = int(scene_id)
        self.aligner = DualCameraStaticAligner(min_inliers=6)
        self.original_aligner = Sim3Aligner(min_inliers=6)
        self.previous = None; self.previous_world = {}
        self.previous_original = None; self.previous_original_world = {}

    def step(self, images, timestamp):
        started = time.perf_counter()
        t = time.perf_counter()
        self.detector_cache.set_context(self.scene_id, int(timestamp))
        detections = self.detector_cache.infer(images)
        detector_s = time.perf_counter() - t
        t = time.perf_counter()
        frame = self.omega.reconstruct(images, int(timestamp))
        omega_wall_s = time.perf_counter() - t
        DetectionFirstWorldMOT._set_dynamic_masks(frame, detections)
        t = time.perf_counter()
        if self.previous is None:
            align = self.aligner.initialize()
            original = self.original_aligner.initialize()
        else:
            old = copy.copy(self.previous); old.world_pointmaps = self.previous_world
            align = self.aligner.update(old, frame, sorted(frame.pointmaps))
            if self.previous_original is None:
                # A failed historical G0-A update must not dereference a
                # missing frame or pretend that the current frame is aligned.
                original = self.original_aligner.update(np.empty((0, 3)), np.empty((0, 3)))
            else:
                old_original = copy.copy(self.previous_original)
                old_original.pointmaps = self.previous_original_world
                common = sorted(set(frame.pointmaps) & set(old_original.pointmaps))
                if common:
                    src, dst = OnlineWorldMOT._static_correspondences(old_original, frame, common[0])
                    original = self.original_aligner.update(src, dst)
                else:
                    original = self.original_aligner.update(np.empty((0, 3)), np.empty((0, 3)))
        alignment_s = time.perf_counter() - t
        aligned = bool(align.result.success and frame.status == "OK")
        transform = align.result.transform
        t = time.perf_counter()
        rows = []
        observations=live_observations(detections,frame,WorldFrameTransform(transform,int(timestamp),aligned))
        for obs in observations:
            support=obs.spatial_support
            rows.append({"view":obs.camera_id,"detection_id":obs.detection_id,
                         "bbox":obs.bbox_xywh.astype(np.float32),"score":obs.score,
                         "box":support.box_points.astype(np.float32) if support else np.empty((0,3),np.float32),
                         "foreground":support.foreground_points.astype(np.float32) if support else np.empty((0,3),np.float32),
                         "center":obs.world_center.astype(np.float32),"cov":obs.world_covariance.astype(np.float32),
                         "quality":obs.geometry_reliability,"foreground_fraction":support.foreground_fraction if support else 0.,
                         "ambiguity":support.ambiguity if support else "WORLD_ALIGNMENT_FAILED",
                         "roi_feature":obs.feature,"feature_valid":obs.feature_valid,
                         "geometry_valid":obs.geometry_valid,"geometry_status":obs.geometry_status})
        if aligned:
            self.previous = frame
            self.previous_world = {v: transform.apply(pm.reshape(-1, 3)).reshape(pm.shape)
                                   for v, pm in frame.pointmaps.items()}
        if original.success:
            self.previous_original = frame
            self.previous_original_world = {v: original.transform.apply(pm.reshape(-1, 3)).reshape(pm.shape)
                                            for v, pm in frame.pointmaps.items()}
        support_s = time.perf_counter() - t
        runtime = {"detector_s": float(detector_s),
                   "omega_wall_s": float(omega_wall_s),
                   "omega_model_only_s": float(frame.diagnostics.get("model_only_s", omega_wall_s)),
                   "omega_preprocess_s": float(frame.diagnostics.get("preprocess_s", 0.0)),
                   "alignment_s": float(alignment_s), "support_s": float(support_s),
                   "total_cache_step_s": float(time.perf_counter() - started)}
        runtime["camera_models"]={str(v):c.to_dict() for v,c in frame.cameras.items()}
        runtime["raw_candidate_shard_sha256"]=self.detector_cache.last_diagnostics["raw_candidate_shard_sha256"]
        runtime["rgb_sha256"]=self.detector_cache.last_diagnostics["rgb_sha256"]
        return rows, frame, detections, align, original, aligned, runtime

    def restore(self, images, timestamp):
        raise RuntimeError("V2_RESUME_REQUIRES_ALIGNMENT_SNAPSHOT: use a fresh output; a single frame cannot restore the world gauge")


def _claim_scene(queue: Path) -> int | None:
    for pending in sorted(queue.glob("scene_*.pending")):
        claim = pending.with_suffix(".running")
        try:
            os.replace(pending, claim)
        except FileNotFoundError:
            continue
        return int(pending.stem.split("_")[1])
    return None


def _fixed_points(rows, key, width=64):
    out = np.full((len(rows), width, 3), np.nan, dtype=np.float32)
    counts = np.zeros(len(rows), dtype=np.int16)
    for i, row in enumerate(rows):
        p = np.asarray(row[key], dtype=np.float32).reshape(-1, 3); n = min(width, len(p))
        if n: out[i, :n] = p[:n]; counts[i] = n
    return out, counts


def cache_scene(ds, sid, output, omega, detector_cache, split, shard_frames=128, limit=None):
    dest = Path(output) / f"scene_{int(sid):02d}"; dest.mkdir(parents=True, exist_ok=True)
    if (dest/"manifest.json").exists():
        raise FileExistsError("V2 cache is immutable; choose a fresh output directory")
    pairs = list(ds.synchronized_pairs(int(sid))); pairs = pairs[:int(limit)] if limit else pairs
    expected = [int(p.frame_id) for p in pairs]
    state = SceneState(omega, detector_cache, sid)
    completed, entries, totals = [], [], collections.Counter()
    for path in sorted(dest.glob("shard_*.json")):
        entry = json.loads(path.read_text()); payload = dest / entry["file"]
        if entry.get("schema") != SCHEMA or not payload.is_file() or sha256(payload) != entry["sha256"]:
            raise RuntimeError(f"invalid WorldObject shard {path}")
        frames = [int(x) for x in entry["timestamps"]]
        if frames != expected[len(completed):len(completed) + len(frames)]:
            raise RuntimeError("non-contiguous WorldObject cache prefix")
        completed.extend(frames); entries.append(entry); totals.update(entry.get("counts", {}))
    if completed:
        last = next(p for p in pairs if int(p.frame_id) == completed[-1])
        state.restore(read_images(ds, last), last.frame_id)
    buffer = []; started = time.perf_counter()

    def flush():
        nonlocal buffer
        if not buffer: return
        offsets = [0]; timestamps=[]; cameras=[]; det_ids=[]; bboxes=[]; scores=[]; centers=[]; covs=[]
        qualities=[]; fg_frac=[]; ambiguity=[]; box_all=[]; fg_all=[]; features=[]; feature_valid=[]; geometry_valid=[]; geometry_status=[]; counts=collections.Counter()
        for pair, rows, align, original, detections, runtime in buffer:
            timestamps.append(int(pair.frame_id)); counts["timestamps"] += 1
            for row in rows:
                cameras.append(row["view"]); det_ids.append(row["detection_id"]); bboxes.append(row["bbox"]); scores.append(row["score"])
                centers.append(row["center"]); covs.append(row["cov"]); qualities.append(row["quality"]); fg_frac.append(row["foreground_fraction"])
                ambiguity.append(row["ambiguity"]); box_all.append(row["box"]); fg_all.append(row["foreground"])
                features.append(row["roi_feature"]);feature_valid.append(row["feature_valid"])
                geometry_valid.append(row["geometry_valid"]);geometry_status.append(row["geometry_status"])
                counts["observations"] += 1
            offsets.append(len(cameras))
        box, box_count = _fixed_points([{"p": x} for x in box_all], "p")
        fg, fg_count = _fixed_points([{"p": x} for x in fg_all], "p")
        # The arrays are intentionally compact; per-frame align details stay
        # in the JSON sidecar where they are auditable.
        ambiguity_codes = {"NONE": 0, "DEPTH_CLUSTERED": 1, "DEPTH_NOT_SEPARATED": 2,
                           "BOX_ONLY": 3, "LOW_SUPPORT": 4, "NO_FINITE_SUPPORT": 5}
        arrays = {"frame_offsets": np.asarray(offsets, np.int64), "timestamps": np.asarray(timestamps, np.int32),
                  "camera_ids": np.asarray(cameras, np.int8), "detection_ids": np.asarray(det_ids, np.int32),
                  "bbox": np.asarray(bboxes, np.float32).reshape(-1, 4), "scores": np.asarray(scores, np.float32),
                  "center": np.asarray(centers, np.float32).reshape(-1, 3), "covariance": np.asarray(covs, np.float32).reshape(-1, 3, 3),
                  "box_support": box, "box_support_count": box_count, "foreground_support": fg, "foreground_support_count": fg_count,
                  "quality": np.asarray(qualities, np.float32), "foreground_fraction": np.asarray(fg_frac, np.float32),
                  "ambiguity": np.asarray([ambiguity_codes.get(str(x), -1) for x in ambiguity], np.int8),
                  "roi_features": np.asarray(features,np.float32).reshape(-1,detector_cache.feature_dim),
                  "feature_valid":np.asarray(feature_valid,bool),"geometry_valid":np.asarray(geometry_valid,bool),
                  "geometry_status":np.asarray(geometry_status,dtype="U48")}
        path = dest / f"shard_{len(entries):04d}.npz"; tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as f: np.savez_compressed(f, **arrays)
        os.replace(tmp, path)
        side = {"schema": SCHEMA, "scene": int(sid), "timestamps": timestamps, "file": path.name,
                "sha256": sha256(path), "frames": len(timestamps), "counts": dict(counts),
                "records": [{"frame_id": int(pair.frame_id), "alignment": {
                    "success": bool(a.result.success), "reason": a.result.reason,
                    "correspondences": int(a.result.correspondences), "inliers": int(a.result.inliers),
                    "inlier_ratio": safe_float(a.result.inlier_ratio), "residual_median": safe_float(a.result.residual_median),
                    "residual_p95": safe_float(a.result.residual_p95), "scale": safe_float(a.result.transform.scale),
                    "roundtrip_error": safe_float(a.roundtrip_error), "used_cameras": list(a.used_cameras),
                    "per_camera": {str(k): {"success": bool(v.success), "inliers": int(v.inliers),
                                               "inlier_ratio": safe_float(v.inlier_ratio), "residual_p95": safe_float(v.residual_p95)}
                                   for k, v in a.per_camera.items()}, "rejected_cameras": a.rejected_cameras},
                    "original_view_alignment": {"success": bool(o.success), "reason": o.reason,
                        "correspondences": int(o.correspondences), "inliers": int(o.inliers),
                        "inlier_ratio": safe_float(o.inlier_ratio), "residual_median": safe_float(o.residual_median),
                        "residual_p95": safe_float(o.residual_p95), "scale": safe_float(o.transform.scale)},
                    "camera_ids": sorted(int(v) for v in pair.images),
                    "image_ids": {str(v): int(rec.image_id) for v, rec in pair.images.items()}}
                           for pair, _, a, o, _, rt in buffer]}
        for record, (_, _, _, _, _, runtime) in zip(side["records"], buffer):
            record["camera_models"]=runtime["camera_models"]
            record["raw_candidate_shard_sha256"]=runtime["raw_candidate_shard_sha256"]
            record["rgb_sha256"]=runtime["rgb_sha256"]
            record["runtime"] = {k:v for k,v in runtime.items() if k.endswith("_s")}
        atomic_json(path.with_suffix(".json"), side); entries.append(side); totals.update(counts); completed.extend(timestamps); buffer=[]
        atomic_json(dest / "progress.json", {"scene": int(sid), "done": len(completed), "expected": len(expected),
                                               "counts": dict(totals), "elapsed_s": time.perf_counter() - started})
        print(json.dumps({"scene": int(sid), "done": len(completed), "expected": len(expected), "observations": totals["observations"]}), flush=True)

    for pair in pairs[len(completed):]:
        images = read_images(ds, pair); rows, frame, detections, align, original, aligned, runtime = state.step(images, pair.frame_id)
        buffer.append((pair, rows, align, original, detections, runtime))
        if len(buffer) >= int(shard_frames): flush()
    flush()
    atomic_json(dest / "manifest.json", {"schema": SCHEMA, "scene": int(sid),
        "split":split,
        "status": "COMPLETE" if len(completed) == len(expected) else "PARTIAL",
        "timestamps": len(completed), "expected_timestamps": len(expected),
        "shards": [{"file": x["file"], "sha256": x["sha256"], "frames": x["frames"],
                    "sidecar_sha256":sha256(dest/Path(x["file"]).with_suffix(".json"))} for x in entries],
        "counts": dict(totals), "raw_gt_modified": False, "gt_in_cache": False,
        "detector_cache": detector_cache.manifest(), "omega": omega.diagnostics(),
        "feature_dim":detector_cache.feature_dim,"source_commit":source_commit(),"source_sha256":source_fingerprint(),
        "support_contract": "camera_forward_depth_to_persistent_world_v2"})


def worker(a):
    require_experiment_authorization(a.authorization,"cache",a.split,"FULL_WORLD_OBJECT_V3")
    import torch
    torch.set_num_threads(2)
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:
        pass
    ds = VisionTrack(a.dataset_root, split=a.split, load_annotations=False)
    detector_cache = ImmutableGMTDetectionCache(a.raw_cache)
    omega = OmegaGeometryBackend(a.omega_source, a.omega_weights, device=a.device, expose_intermediate_features=False)
    if omega.status != "READY": raise RuntimeError(f"Omega unavailable: {omega.reason}")
    while True:
        sid = _claim_scene(Path(a.scene_queue)) if a.scene_queue else None
        if sid is None:
            break
        try:
            cache_scene(ds, sid, a.output, omega, detector_cache, a.split, a.shard_frames, a.limit)
            claim = Path(a.scene_queue) / f"scene_{sid:02d}.running"
            claim.unlink(missing_ok=True)
        except Exception:
            claim = Path(a.scene_queue) / f"scene_{sid:02d}.failed"
            claim.write_text("failed")
            raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--authorization")
    p.add_argument("--dataset-root", default="/data/DATASETS/TRACKING/JDE/VisionTrack")
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--output", required=True); p.add_argument("--raw-cache", required=True)
    p.add_argument("--scenes", nargs="*", type=int); p.add_argument("--scene-queue")
    p.add_argument("--device", default="cuda:0"); p.add_argument("--shard-frames", type=int, default=128)
    p.add_argument("--limit", type=int); p.add_argument("--omega-source", default=DEFAULT_OMEGA); p.add_argument("--omega-weights", default=DEFAULT_OMEGA_WEIGHTS)
    a = p.parse_args(); np.set_printoptions(suppress=True)
    require_experiment_authorization(a.authorization,"cache",a.split,"FULL_WORLD_OBJECT_V3")
    import torch
    torch.set_num_threads(2)
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:
        pass
    ds = VisionTrack(a.dataset_root, split=a.split, load_annotations=False)
    if a.scene_queue: worker(a); return
    if not a.scenes: raise SystemExit("--scenes or --scene-queue is required")
    detector_cache = ImmutableGMTDetectionCache(a.raw_cache)
    omega = OmegaGeometryBackend(a.omega_source, a.omega_weights, device=a.device, expose_intermediate_features=False)
    if omega.status != "READY": raise RuntimeError(f"Omega unavailable: {omega.reason}")
    for sid in a.scenes: cache_scene(ds, sid, a.output, omega, detector_cache, a.split, a.shard_frames, a.limit)


if __name__ == "__main__": main()
