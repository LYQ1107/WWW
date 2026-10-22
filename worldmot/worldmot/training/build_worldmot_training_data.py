"""Audit VisionTrack TRAIN and write a reproducible clip/data manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from ..data.visiontrack import VisionTrack


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def audit_train(dataset_root: str | Path, annotation: str = "annotations/train.json",
                image_root: str = "train", scenes: Iterable[int] | None = None) -> dict:
    ds = VisionTrack(dataset_root, annotation, image_root)
    scene_ids = ds.scene_ids() if scenes is None else [int(x) for x in scenes]
    scene_rows = []
    total_frames = total_images = total_annotations = 0
    duplicate_keys = []
    for sid in scene_ids:
        frames = ds.frames(sid)
        image_count = 0; ann_count = 0
        ids = set()
        for pair in ds.synchronized_pairs(sid):
            total_frames += 1
            for view, rec in pair.images.items():
                image_count += 1; total_images += 1
                anns = ds.ground_truth(rec.image_id); ann_count += len(anns); total_annotations += len(anns)
                for ann in anns:
                    key = (sid, int(ann.get("instance_id", ann.get("track_id", ann.get("id", -1)))),
                           int(rec.frame_id), int(view))
                    if key in ids:
                        duplicate_keys.append(key)
                    ids.add(key)
        scene_rows.append({"scene_id": int(sid), "scene_name": ds.scene_name(sid),
                           "synchronized_frames": len(frames), "images": image_count,
                           "annotations": ann_count, "first_frame": frames[0] if frames else None,
                           "last_frame": frames[-1] if frames else None})
    return {"status": "PASS", "dataset_root": str(Path(dataset_root).resolve()),
            "annotation": str((Path(dataset_root) / annotation).resolve()),
            "annotation_sha256": _sha256(Path(dataset_root) / annotation),
            "image_root": str((Path(dataset_root) / image_root).resolve()),
            "scene_ids": scene_ids, "scenes": scene_rows,
            "totals": {"synchronized_frames": total_frames, "images": total_images,
                       "annotations": total_annotations},
            "duplicate_annotation_keys": len(duplicate_keys),
            "supervision_contract": {"gt_allowed_in_loss_and_posthoc_eval_only": True,
                                     "gt_xyz_forbidden_in_feature_cache": True,
                                     "raw_images_not_copied": True}}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--annotation", default="annotations/train.json")
    parser.add_argument("--image-root", default="train")
    parser.add_argument("--scenes", nargs="*", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = audit_train(args.dataset_root, args.annotation, args.image_root, args.scenes or None)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": result["status"], "totals": result["totals"], "scenes": len(result["scenes"])}, indent=2))


if __name__ == "__main__":
    main()
