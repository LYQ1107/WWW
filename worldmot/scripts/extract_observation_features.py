#!/usr/bin/env python3
"""Build or audit leakage-safe object-level PWQT feature caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def audit(path: str | Path) -> dict:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        forbidden = sorted({"gt_xyz", "ground_truth_xyz", "oracle_xyz"} & set(data.files))
        required = sorted({"observations", "geometry", "assignment", "birth", "visibility"} - set(data.files))
        if forbidden:
            raise ValueError(f"forbidden GT fields in feature cache: {forbidden}")
        return {"status": "PASS" if not required else "INVALID", "path": str(path.resolve()),
                "keys": sorted(data.files), "missing": required,
                "rows": int(data["observations"].shape[0]) if "observations" in data else 0,
                "gt_geometry_forbidden": True}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def extract_clip(dataset_root: str | Path, scene: int, limit: int, output: str | Path,
                 detector_config: str | Path, detector_checkpoint: str | Path,
                 omega_source: str | Path, omega_checkpoint: str | Path,
                 detector_device: str = "cuda:3", omega_device: str = "cuda:2",
                 max_world_queries: int = 128) -> dict:
    """Extract a short registered clip; GT is read only for target creation."""
    from PIL import Image

    from worldmot.data.visiontrack import VisionTrack
    from worldmot.geometry.world_alignment import Sim3Aligner
    from worldmot.models.omega_geometry_backend import OmegaGeometryBackend
    from worldmot.models.person_detector import PersonDetector
    from worldmot.models.world_observation_encoder import MultiViewObservationEncoder
    from worldmot.runtime.online_worldmot import OnlineWorldMOT
    from worldmot.runtime.online_worldmot_v2 import OnlineWorldMOTV2
    from worldmot.training.identity_targets import match_detections_to_gt

    root = Path(dataset_root).resolve()
    ds = VisionTrack(root)
    pairs = list(ds.synchronized_pairs(int(scene), limit=int(limit)))
    if not pairs:
        raise RuntimeError(f"no synchronized frames for scene {scene}")
    detector = PersonDetector(detector_config, detector_checkpoint, device=detector_device,
                              expose_intermediate_features=True)
    geometry = OmegaGeometryBackend(omega_source, omega_checkpoint, device=omega_device,
                                    expose_intermediate_features=True)
    if detector.status != "READY":
        raise RuntimeError(f"detector unavailable: {detector.reason}")
    if geometry.status != "READY":
        raise RuntimeError(f"Omega unavailable: {geometry.reason}")
    encoder = MultiViewObservationEncoder()
    aligner = Sim3Aligner()
    previous = None
    previous_world_points = {}
    rows = []
    identity_slots = {}
    seen = set()
    alignment_rows = []
    for pair in pairs:
        images = {int(v): np.asarray(Image.open(ds.image_path(rec)).convert("RGB"))
                  for v, rec in pair.images.items()}
        detections = detector.infer(images)
        frame = geometry.reconstruct(images, pair.frame_id)
        OnlineWorldMOTV2._set_dynamic_masks(frame, detections)
        if previous is None:
            alignment = aligner.initialize()
        else:
            import copy
            old = copy.copy(previous)
            old.pointmaps = previous_world_points or previous.pointmaps
            common = sorted(set(frame.pointmaps) & set(old.pointmaps))
            if common:
                src, dst = OnlineWorldMOT._static_correspondences(old, frame, common[0])
                alignment = aligner.update(src, dst) if len(src) >= aligner.min_inliers else aligner.update(np.zeros((0, 3)), np.zeros((0, 3)))
            else:
                alignment = aligner.update(np.zeros((0, 3)), np.zeros((0, 3)))
        transform = alignment.transform if alignment.success else None
        tokens = encoder.encode(
            detections, images, frame, pair.frame_id, transform,
            detector_feature_maps=detector.last_feature_maps,
            omega_patch_features=frame.omega_patch_features,
        )
        gt = ds.ground_truth_pair(pair)
        target_map = match_detections_to_gt(detections, gt, pair.scene_id, iou_threshold=0.5)
        rows.append((tokens, target_map, alignment))
        alignment_rows.append({"frame_id": int(pair.frame_id), "success": bool(alignment.success),
                               "reason": alignment.reason, "inliers": int(alignment.inliers)})
        if alignment.success:
            previous = frame
            previous_world_points = {v: alignment.transform.apply(x.reshape(-1, 3)).reshape(x.shape)
                                     for v, x in frame.pointmaps.items()}
    max_obs = max(len(tokens) for tokens, _, _ in rows)
    obs_array = np.zeros((len(rows), max_obs, 340), dtype=np.float32)
    geo_array = np.zeros((len(rows), max_obs, 18), dtype=np.float32)
    camera_array = np.zeros((len(rows), max_obs), dtype=np.int64)
    valid_array = np.zeros((len(rows), max_obs), dtype=bool)
    assignment_array = np.zeros((len(rows), max_world_queries, max_obs + 1), dtype=np.int8)
    birth_array = np.zeros((len(rows), max_obs), dtype=np.float32)
    visibility_array = np.zeros((len(rows), max_world_queries, 2), dtype=np.float32)
    overflow = []
    for row_idx, (tokens, target_map, _) in enumerate(rows):
        assignment_array[row_idx, :, -1] = 1
        local_new = set()
        for obs_idx, token in enumerate(tokens):
            obs_array[row_idx, obs_idx] = token.feature
            geo_array[row_idx, obs_idx] = token.geometry_feature
            camera_array[row_idx, obs_idx] = int(token.camera_id)
            valid_array[row_idx, obs_idx] = bool(token.valid)
            target = target_map.get((int(token.camera_id), int(token.detection_id)))
            key = target.identity_key if target is not None else None
            if key is None:
                continue
            if key not in identity_slots:
                identity_slots[key] = len(identity_slots)
            slot = identity_slots[key]
            if slot >= max_world_queries:
                overflow.append(key)
                continue
            assignment_array[row_idx, slot, -1] = 0
            assignment_array[row_idx, slot, obs_idx] = 1
            camera_index = 0 if int(token.camera_id) == 1 else 1
            visibility_array[row_idx, slot, camera_index] = 1
            if key not in seen and key not in local_new:
                birth_array[row_idx, obs_idx] = 1
                local_new.add(key)
        seen.update(local_new)
    out = Path(output).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, observations=obs_array, geometry=geo_array,
                        observation_valid=valid_array, camera_ids=camera_array,
                        assignment=assignment_array,
                        birth=birth_array, visibility=visibility_array)
    manifest = {
        "status": "PASS_REAL_OBJECT_FEATURE_CACHE",
        "scene": int(scene), "frames": [int(p.frame_id) for p in pairs],
        "rows": len(rows), "max_observations": int(max_obs),
        "identity_slots": identity_slots, "query_capacity_overflow": sorted(set(overflow)),
        "cache": str(out), "cache_sha256": _sha256(out),
        "detector": detector.manifest(), "omega": geometry.diagnostics(),
        "alignment": alignment_rows,
        "supervision": {"gt_used_for_targets_only": True, "gt_xyz_in_cache": False,
                         "raw_images_in_cache": False},
    }
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")
    check = sub.add_parser("audit")
    check.add_argument("cache"); check.add_argument("--output")
    build = sub.add_parser("extract")
    build.add_argument("--dataset-root", required=True); build.add_argument("--scene", type=int, required=True)
    build.add_argument("--limit", type=int, default=3); build.add_argument("--output", required=True)
    build.add_argument("--detector-config", required=True); build.add_argument("--detector-checkpoint", required=True)
    build.add_argument("--omega-source", required=True); build.add_argument("--omega-checkpoint", required=True)
    build.add_argument("--detector-device", default="cuda:3"); build.add_argument("--omega-device", default="cuda:2")
    build.add_argument("--max-world-queries", type=int, default=128)
    args = p.parse_args(argv)
    if args.command == "extract":
        result = extract_clip(args.dataset_root, args.scene, args.limit, args.output,
                              args.detector_config, args.detector_checkpoint,
                              args.omega_source, args.omega_checkpoint,
                              args.detector_device, args.omega_device, args.max_world_queries)
    elif args.command == "audit":
        result = audit(args.cache)
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        p.error("choose audit or extract")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

