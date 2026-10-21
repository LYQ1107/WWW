"""Extract a matched frozen DINOv2 control for all causal detector boxes."""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DINO_SOURCE = Path("/data1/liuyeqiang/WWW_vggt_geo_runtime/sources/dinov2")
VGGT_SOURCE = Path("/data1/liuyeqiang/WWW_vggt_geo_runtime/sources/vggt")
DINO_CHECKPOINT = Path("/data1/liuyeqiang/WWW_vggt_geo_runtime/weights/dinov2_vitl14_reg4.pth")
RUNTIME_AUDIT = Path("/data1/liuyeqiang/WWW_omega_fulltrain_runtime_20261016/full_train_audit_v1")
PRIOR_AUDIT = Path("/data1/liuyeqiang/WWW_trackvggt_runtime/full_train_audit_v1")
DATA = Path("/data/DATASETS/TRACKING/JDE/VisionTrack")
TARGET = 518
PATCH = 14
FEATURE_DIM = 1024


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit_root(scene):
    fresh = RUNTIME_AUDIT / f"scene{scene}_full_v1"
    return fresh if fresh.is_dir() else PRIOR_AUDIT / f"scene{scene}_full_v1"


def _detectors(scene):
    audit = _audit_root(scene)
    detector = {}
    with gzip.open(audit / "detector_provenance.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            key = tuple(int(value) for value in row["source"])
            if key in detector:
                raise ValueError(f"duplicate detector key {key}")
            detector[key] = tuple(float(value) for value in row["detection"]["bbox"])
    return audit, detector


def _rows(scene):
    data = json.loads((DATA / "annotations/train.json").read_text())
    rows = [dict(row, image_id=int(row["id"]), file_name=str(DATA / "train" / row["file_name"])) for row in data["images"] if int(row["video_id"]) == scene]
    rows.sort(key=lambda row: (int(row["frame_id"]), int(row["view_id"]), int(row["image_id"])))
    groups = collections.defaultdict(list)
    for row in rows:
        groups[int(row["frame_id"])].append(row)
    if sorted(groups) != list(range(1, len(groups) + 1)) or any(len(value) != 2 for value in groups.values()):
        raise ValueError(f"scene {scene} is not a complete two-view video")
    return rows, dict(sorted(groups.items()))


def _mapping_box(box):
    x, y, width, height = box
    resized_h = round(1080 * TARGET / 1920 / PATCH) * PATCH
    offset_y = (TARGET - resized_h) // 2
    scale_x = TARGET / 1920.0
    scale_y = resized_h / 1080.0
    return (float(np.clip(x * scale_x, 0, TARGET)), float(np.clip(y * scale_y + offset_y, 0, TARGET)),
            float(np.clip((x + width) * scale_x, 0, TARGET)), float(np.clip((y + height) * scale_y + offset_y, 0, TARGET)))


def _load_model(device):
    sys.path[:0] = [str(DINO_SOURCE), str(VGGT_SOURCE)]
    import torch
    from dinov2.hub.backbones import dinov2_vitl14_reg

    model = dinov2_vitl14_reg(pretrained=False).eval().to(device)
    state = torch.load(DINO_CHECKPOINT, map_location="cpu")
    model.load_state_dict(state, strict=True)
    del state
    return model


def _extract_scene(scene, output_root, physical_device, batch_size):
    sys.path[:0] = [str(DINO_SOURCE), str(VGGT_SOURCE)]
    import torch
    import torch.nn.functional as F
    from torchvision.ops import roi_align
    from vggt.utils.load_fn import load_and_preprocess_images

    output = output_root / f"scene{scene}_dinov2_v1"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    feature_path = output / "features.npz"
    if manifest_path.exists() and feature_path.exists() and json.loads(manifest_path.read_text()).get("status") == "COMPLETE_DINOV2_FEATURE_EXTRACTION":
        print(json.dumps({"scene": scene, "status": "SKIP_EXISTING"}), flush=True)
        return
    if manifest_path.exists() or feature_path.exists():
        raise FileExistsError(f"partial DINO output exists: {output}")
    audit, detector = _detectors(scene)
    rows, groups = _rows(scene)
    image_sources = collections.defaultdict(list)
    for key, box in detector.items():
        image_sources[key[0]].append((key[1], box))
    for image_id in image_sources:
        image_sources[image_id].sort()
    detector_keys = sorted(detector)
    key_index = {key: index for index, key in enumerate(detector_keys)}
    features = np.full((len(detector_keys), FEATURE_DIM), np.nan, dtype=np.float16)
    boxes = np.asarray([_mapping_box(detector[key]) for key in detector_keys], dtype=np.float32)
    device = torch.device("cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else f"cuda:{physical_device}")
    model = _load_model(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    height = width = TARGET
    patch_height = patch_width = TARGET // PATCH
    frame_items = list(groups.items())
    started = time.perf_counter()
    completed = 0
    for start in range(0, len(frame_items), batch_size):
        batch = frame_items[start : start + batch_size]
        images = torch.stack([load_and_preprocess_images([row["file_name"] for row in group], mode="pad") for _, group in batch]).to(device)
        if tuple(images.shape[-2:]) != (TARGET, TARGET):
            raise ValueError(f"unexpected DINO preprocessing shape {tuple(images.shape)}")
        flat = images.reshape(len(batch) * 2, 3, TARGET, TARGET)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            tokens = model.forward_features((flat - mean) / std)["x_norm_patchtokens"]
            if tokens.shape[1] != patch_height * patch_width or tokens.shape[2] != FEATURE_DIM:
                raise ValueError(f"unexpected DINO token shape {tuple(tokens.shape)}")
            fmap = tokens.reshape(len(batch) * 2, patch_height, patch_width, FEATURE_DIM).permute(0, 3, 1, 2).float()
            roi_boxes = []
            roi_keys = []
            for batch_index, (_, group) in enumerate(batch):
                for view_index, row in enumerate(group):
                    for det_index, _ in image_sources[int(row["image_id"])]:
                        key = (int(row["image_id"]), int(det_index))
                        roi_boxes.append([batch_index * 2 + view_index, *boxes[key_index[key]]])
                        roi_keys.append(key)
            if roi_boxes:
                pooled = roi_align(fmap, torch.tensor(roi_boxes, dtype=torch.float32, device=device), output_size=(1, 1), spatial_scale=1.0 / PATCH, aligned=True)
                pooled = F.normalize(pooled[:, :, 0, 0], dim=1, eps=1e-8).cpu().numpy().astype(np.float16)
                for key, vector in zip(roi_keys, pooled):
                    target = key_index[key]
                    if np.isfinite(features[target]).any():
                        raise ValueError(f"duplicate DINO feature assignment for {key}")
                    features[target] = vector
        completed += len(batch)
        if completed == len(frame_items) or completed % (batch_size * 20) == 0:
            elapsed = time.perf_counter() - started
            print(json.dumps({"scene": scene, "groups": completed, "total_groups": len(frame_items), "seconds": elapsed, "groups_per_second": completed / max(elapsed, 1e-9)}), flush=True)
    if not np.isfinite(features.astype(np.float32)).all():
        raise ValueError("DINO feature store has unassigned/non-finite values")
    np.savez(feature_path, source_keys=np.asarray(detector_keys, dtype=np.int64), features=features, boxes_xyxy_518=boxes)
    manifest = {
        "status": "COMPLETE_DINOV2_FEATURE_EXTRACTION", "scene_id": scene, "audit_root": str(audit),
        "audit_manifest_sha256": _sha256(audit / "manifest.json"), "detector_source_count": len(detector_keys),
        "groups": len(frame_items), "images": len(rows), "feature_dim": FEATURE_DIM, "feature_dtype": "float16",
        "feature_definition": "frozen DINOv2 ViT-L/14 register patch tokens ROIAlign 1x1, L2 normalized",
        "checkpoint": str(DINO_CHECKPOINT), "checkpoint_sha256": _sha256(DINO_CHECKPOINT),
        "source_roots": {"dinov2": str(DINO_SOURCE), "vggt_load_fn": str(VGGT_SOURCE)},
        "preprocess": {"mode": "pad", "image_resolution": TARGET, "patch_size": PATCH, "input_shape": [3, TARGET, TARGET], "original_shape": [3, 1080, 1920]},
        "device_index": physical_device, "batch_size": batch_size, "wall_seconds": time.perf_counter() - started,
        "ground_truth_opened": False, "features_sha256": _sha256(feature_path), "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"scene": scene, "status": manifest["status"], "detector_sources": len(detector_keys), "seconds": manifest["wall_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-ids", nargs="+", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if any(scene < 1 or scene > 24 for scene in args.scene_ids) or not 1 <= args.batch_size <= 32:
        raise ValueError("invalid scene or batch size")
    if not DINO_CHECKPOINT.is_file():
        raise FileNotFoundError(DINO_CHECKPOINT)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_index)
    for scene in args.scene_ids:
        _extract_scene(scene, args.output_root, args.device_index, args.batch_size)


if __name__ == "__main__":
    main()
