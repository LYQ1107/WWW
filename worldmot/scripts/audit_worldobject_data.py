#!/usr/bin/env python3
"""Audit VisionTrack data and freeze the new WorldObject experiment inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from worldmot.data.visiontrack import VisionTrack


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def audit_split(root: Path, split: str) -> dict:
    ds = VisionTrack(root, split=split, load_annotations=True)
    scenes = []
    duplicate_keys = Counter()
    for sid in ds.scene_ids():
        frames = ds.frames(sid)
        anns = 0
        for pair in ds.synchronized_pairs(sid):
            for view, rec in pair.images.items():
                rows = ds.ground_truth(rec.image_id)
                anns += len(rows)
                duplicate_keys.update((int(sid), int(a.get("instance_id", a.get("track_id", a.get("id", -1))))) for a in rows)
        scenes.append({"scene_id": int(sid), "scene_name": ds.scene_name(sid),
                       "timestamps": len(frames), "first_frame": frames[0] if frames else None,
                       "last_frame": frames[-1] if frames else None, "gt_boxes": anns})
    return {"split": split, "annotation": str(ds.annotation_path),
            "annotation_sha256": sha256(ds.annotation_path), "scene_count": len(scenes),
            "timestamps": sum(x["timestamps"] for x in scenes),
            "images": sum(2 * x["timestamps"] for x in scenes),
            "scenes": scenes,
            "duplicate_scene_identity_keys": sum(v > 1 for v in duplicate_keys.values()),
            "duplicate_key_examples": [list(k) + [v] for k, v in duplicate_keys.items() if v > 1][:32]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="/data/DATASETS/TRACKING/JDE/VisionTrack")
    p.add_argument("--output", required=True)
    p.add_argument("--metadata", nargs="*", default=[])
    a = p.parse_args()
    root = Path(a.dataset_root).resolve()
    result = {"schema": "worldobject_data_audit_v1", "dataset_root": str(root),
              "dataset_mount": os.path.ismount(root.parent),
              "splits": {s: audit_split(root, s) for s in ("train", "test")}}
    result["metadata"] = [{"path": str(Path(x).resolve()), "sha256": sha256(Path(x))}
                          for x in a.metadata if Path(x).is_file()]
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp"); tmp.write_text(json.dumps(result, indent=2, sort_keys=True)); os.replace(tmp, out)
    print(json.dumps({"output": str(out), "train_scenes": result["splits"]["train"]["scene_count"],
                      "test_scenes": result["splits"]["test"]["scene_count"],
                      "train_timestamps": result["splits"]["train"]["timestamps"],
                      "test_timestamps": result["splits"]["test"]["timestamps"]}, indent=2))


if __name__ == "__main__":
    main()
