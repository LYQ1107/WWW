"""Extract frozen VGGT-Ω 512 patch features for every causal GMT detection.

The runner consumes only the detector provenance and the actual causal GMT
decision streams.  It never opens VisionTrack ground truth.  Images are grouped
by their synchronized frame and both camera views are passed to the official
VGGT-Ω aggregator together, so the saved ROI vectors are produced by the
released 512 model rather than by the older VGGT adapter in this repository.

The output is intentionally an external runtime artifact: one uncompressed
``features.npz`` per scene contains ``source_keys`` (image id, detection
index), ``features`` (float16, final Ω patch token, 2048 dimensions), and
``boxes_xyxy_512``.  The manifest records the checkpoint and preprocessing
contract.  A missing or malformed source stops the scene instead of silently
dropping a candidate.
"""

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
OMEGA_SOURCE = Path("/data1/liuyeqiang/WWW_omega_runtime_20261016/official_vggt_omega")
OMEGA_CHECKPOINT = Path("/data1/liuyeqiang/WWW_omega_runtime_20261016/weights/vggt_omega_1b_512.pt")
RUNTIME_AUDIT = Path("/data1/liuyeqiang/WWW_omega_fulltrain_runtime_20261016/full_train_audit_v1")
PRIOR_AUDIT = Path("/data1/liuyeqiang/WWW_trackvggt_runtime/full_train_audit_v1")
DATA = Path("/data/DATASETS/TRACKING/JDE/VisionTrack")
PATCH_SIZE = 16
IMAGE_RESOLUTION = 512
FEATURE_DIM = 2048


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit_root(scene_id: int) -> Path:
    fresh = RUNTIME_AUDIT / f"scene{scene_id}_full_v1"
    if fresh.is_dir():
        return fresh
    old = PRIOR_AUDIT / f"scene{scene_id}_full_v1"
    if old.is_dir():
        return old
    raise FileNotFoundError(f"no complete causal audit for scene {scene_id}")


def _read_sources(scene_id: int):
    """Return detector boxes and the source keys admitted by actual calls."""
    audit = _audit_root(scene_id)
    detector = {}
    with gzip.open(audit / "detector_provenance.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            key = tuple(int(value) for value in row["source"])
            if key in detector:
                raise ValueError(f"duplicate detector source {key} in {audit}")
            box = row["detection"]["bbox"]
            if len(box) != 4 or not all(np.isfinite(float(value)) for value in box):
                raise ValueError(f"invalid detector bbox for {key}: {box}")
            detector[key] = tuple(float(value) for value in box)

    used = set()
    with gzip.open(audit / "actual_decisions.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            used.update(tuple(int(value) for value in source) for source in row.get("queries", []))
            used.update(tuple(int(value) for value in ref["source"]) for ref in row.get("references", []))
    if not used.issubset(detector):
        missing = sorted(used - detector)
        raise ValueError(f"{len(missing)} causal sources have no detector provenance; sample={missing[:5]}")
    # Extract all detector boxes in a frame, not only sources that happened to
    # enter a decision.  This makes the feature store complete for the frozen
    # detector candidate set while the training join can still restrict itself
    # to actual causal queries/references.
    return audit, detector, used


def _load_train_rows(scene_id: int):
    data = json.loads((DATA / "annotations/train.json").read_text())
    rows = [
        dict(row, image_id=int(row["id"]), file_name=str(DATA / "train" / row["file_name"]))
        for row in data["images"]
        if int(row["video_id"]) == scene_id
    ]
    rows.sort(key=lambda row: (int(row["frame_id"]), int(row["view_id"]), int(row["image_id"])))
    if not rows:
        raise ValueError(f"scene {scene_id} has no TRAIN images")
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[int(row["frame_id"])].append(row)
    expected = list(range(1, len(grouped) + 1))
    if sorted(grouped) != expected or any(len(group) != 2 for group in grouped.values()):
        raise ValueError(f"scene {scene_id} is not a complete synchronized two-view video")
    return rows, dict(sorted(grouped.items()))


def _source_to_frame(rows, detector):
    image_by_id = {int(row["image_id"]): row for row in rows}
    image_sources = collections.defaultdict(list)
    for (image_id, det_index), bbox in detector.items():
        if image_id not in image_by_id:
            raise ValueError(f"detector source image {image_id} is outside scene rows")
        image_sources[image_id].append((det_index, bbox))
    for image_id in image_sources:
        image_sources[image_id].sort(key=lambda value: value[0])
    return image_by_id, image_sources


def _scaled_box(bbox):
    # All VisionTrack TRAIN frames are 1920x1080.  The official max_size
    # preprocessing maps them exactly to 512x288 (no aspect crop or padding).
    x, y, width, height = bbox
    x0 = np.clip(x * IMAGE_RESOLUTION / 1920.0, 0.0, IMAGE_RESOLUTION)
    y0 = np.clip(y * (IMAGE_RESOLUTION * 1080.0 / 1920.0) / 1080.0, 0.0, IMAGE_RESOLUTION * 1080.0 / 1920.0)
    x1 = np.clip((x + width) * IMAGE_RESOLUTION / 1920.0, 0.0, IMAGE_RESOLUTION)
    y1 = np.clip((y + height) * (IMAGE_RESOLUTION * 1080.0 / 1920.0) / 1080.0, 0.0, IMAGE_RESOLUTION * 1080.0 / 1920.0)
    if not (x1 > x0 and y1 > y0):
        raise ValueError(f"degenerate scaled box from {bbox}")
    return (float(x0), float(y0), float(x1), float(y1))


def _load_model(device):
    sys.path.insert(0, str(OMEGA_SOURCE))
    import torch
    from vggt_omega.models import VGGTOmega

    model = VGGTOmega().eval().to(device)
    state = torch.load(OMEGA_CHECKPOINT, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        state = state.get("model", state.get("state_dict", state))
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"strict Ω load did not match: {result}")
    del state
    return model


def _extract_scene(scene_id: int, output_root: Path, device_index: int, batch_size: int):
    # Imports are delayed so a CPU-only inventory/audit can still import this
    # module without pulling the official CUDA stack into GMT's environment.
    sys.path.insert(0, str(OMEGA_SOURCE))
    import torch
    from torchvision.ops import roi_align
    from vggt_omega.utils.load_fn import load_and_preprocess_images

    output = output_root / f"scene{scene_id}_omega512_v1"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    feature_path = output / "features.npz"
    if manifest_path.exists() and feature_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("status") == "COMPLETE_OMEGA512_FEATURE_EXTRACTION":
            print(json.dumps({"scene": scene_id, "status": "SKIP_EXISTING", "output": str(output)}), flush=True)
            return
    if manifest_path.exists() or feature_path.exists():
        raise FileExistsError(f"partial Ω feature output exists: {output}")

    audit, detector, used = _read_sources(scene_id)
    rows, groups = _load_train_rows(scene_id)
    image_by_id, image_sources = _source_to_frame(rows, detector)
    detector_keys = sorted(detector)
    key_index = {key: index for index, key in enumerate(detector_keys)}
    features = np.full((len(detector_keys), FEATURE_DIM), np.nan, dtype=np.float16)
    boxes_512 = np.empty((len(detector_keys), 4), dtype=np.float32)
    for index, key in enumerate(detector_keys):
        boxes_512[index] = _scaled_box(detector[key])

    # The launch wrapper pins one physical GPU through CUDA_VISIBLE_DEVICES;
    # inside that process it is always logical device 0.
    device = torch.device("cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else f"cuda:{device_index}")
    model = _load_model(device)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    total_groups = len(groups)
    started = time.perf_counter()
    completed_groups = 0
    height, width = 288, 512
    patch_height, patch_width = height // PATCH_SIZE, width // PATCH_SIZE

    frame_items = list(groups.items())
    for start in range(0, len(frame_items), batch_size):
        batch = frame_items[start : start + batch_size]
        paths = [[row["file_name"] for row in group] for _, group in batch]
        images = torch.stack(
            [load_and_preprocess_images(path_list, mode="max_size", image_resolution=IMAGE_RESOLUTION, patch_size=PATCH_SIZE) for path_list in paths]
        ).to(device, non_blocking=True)
        if tuple(images.shape[-2:]) != (height, width):
            raise ValueError(f"unexpected Ω preprocessing shape {tuple(images.shape)}")
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dtype):
            aggregated, patch_token_start = model.aggregator(images)
            final = aggregated[-1]
            if final is None:
                raise RuntimeError("Ω aggregator did not return the final cached layer")
            patch = final[:, :, patch_token_start:, :]
            if patch.shape[2] != patch_height * patch_width or patch.shape[3] != FEATURE_DIM:
                raise ValueError(f"unexpected Ω patch shape {tuple(patch.shape)}")
            patch = patch.reshape(len(batch) * 2, patch_height, patch_width, FEATURE_DIM).permute(0, 3, 1, 2).float()
            boxes = []
            box_keys = []
            for batch_index, (_, group) in enumerate(batch):
                for view_index, row in enumerate(group):
                    image_id = int(row["image_id"])
                    for det_index, _ in image_sources.get(image_id, []):
                        key = (image_id, int(det_index))
                        boxes.append([batch_index * 2 + view_index, *boxes_512[key_index[key]]])
                        box_keys.append(key)
            if boxes:
                box_tensor = torch.tensor(boxes, dtype=torch.float32, device=device)
                pooled = roi_align(patch, box_tensor, output_size=(1, 1), spatial_scale=1.0 / PATCH_SIZE, aligned=True)
                pooled = pooled[:, :, 0, 0].detach().cpu().numpy().astype(np.float16, copy=False)
                if pooled.shape != (len(box_keys), FEATURE_DIM):
                    raise ValueError(f"unexpected ROI output shape {pooled.shape}")
                for key, vector in zip(box_keys, pooled):
                    target = key_index[key]
                    if np.any(np.isfinite(features[target])):
                        raise ValueError(f"duplicate Ω feature assignment for {key}")
                    features[target] = vector
        completed_groups += len(batch)
        if completed_groups == total_groups or completed_groups % (batch_size * 20) == 0:
            elapsed = time.perf_counter() - started
            print(json.dumps({"scene": scene_id, "groups": completed_groups, "total_groups": total_groups, "seconds": elapsed, "groups_per_second": completed_groups / max(elapsed, 1e-9)}), flush=True)

    if not np.isfinite(features.astype(np.float32)).all():
        raise ValueError("Ω feature store contains an unassigned/non-finite vector")
    np.savez(
        feature_path,
        source_keys=np.asarray(detector_keys, dtype=np.int64),
        features=features,
        boxes_xyxy_512=boxes_512,
    )
    manifest = {
        "status": "COMPLETE_OMEGA512_FEATURE_EXTRACTION",
        "scene_id": scene_id,
        "audit_root": str(audit),
        "audit_manifest_sha256": _sha256(audit / "manifest.json"),
        "causal_used_source_count": len(used),
        "detector_source_count": len(detector_keys),
        "groups": total_groups,
        "images": len(rows),
        "feature_dim": FEATURE_DIM,
        "feature_dtype": "float16",
        "feature_definition": "official VGGT-Omega final cached patch tokens (frame + inter-frame, 2048) ROIAlign 1x1 over detector bbox",
        "checkpoint": str(OMEGA_CHECKPOINT),
        "checkpoint_sha256": _sha256(OMEGA_CHECKPOINT),
        "official_source": str(OMEGA_SOURCE),
        "preprocess": {"mode": "max_size", "image_resolution": IMAGE_RESOLUTION, "patch_size": PATCH_SIZE, "input_shape": [3, 288, 512], "original_shape": [3, 1080, 1920]},
        "device_index": device_index,
        "batch_size": batch_size,
        "wall_seconds": time.perf_counter() - started,
        "future_or_gt_features": False,
        "ground_truth_opened": False,
        "features_sha256": _sha256(feature_path),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"scene": scene_id, "status": manifest["status"], "detector_sources": len(detector_keys), "used_sources": len(used), "seconds": manifest["wall_seconds"], "output": str(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-ids", nargs="+", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if not args.scene_ids or any(scene < 1 or scene > 24 for scene in args.scene_ids):
        raise ValueError("scene IDs must be in 1..24")
    if args.batch_size < 1 or args.batch_size > 16:
        raise ValueError("batch size must be in 1..16")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_index)
    if not OMEGA_CHECKPOINT.is_file():
        raise FileNotFoundError(OMEGA_CHECKPOINT)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for scene_id in args.scene_ids:
        _extract_scene(scene_id, args.output_root, args.device_index, args.batch_size)


if __name__ == "__main__":
    main()
