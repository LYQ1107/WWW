#!/usr/bin/env python3
"""Deterministically choose the A0 three-scene sanity subset.

The selection uses the number of test annotations per video.  For each of the
25th, 50th, and 75th percentiles, the scene with the closest annotation count
is selected; ties are resolved by the scene name.  The original annotation
objects are copied unchanged into a diagnostic subset file.  This is a subset
index, not a GT edit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def quantile(values: list[int], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--subset-json", required=True, type=Path)
    args = ap.parse_args()

    data = json.loads(args.input.read_text())
    images = {x["id"]: x for x in data["images"]}
    counts = {v["id"]: 0 for v in data["videos"]}
    for ann in data["annotations"]:
        counts[images[ann["image_id"]]["video_id"]] += 1

    rows = []
    for video in data["videos"]:
        rows.append(
            {
                "scene": video["file_name"],
                "video_id": video["id"],
                "annotation_count": counts[video["id"]],
                "view_num": video.get("view_num"),
            }
        )
    values = [x["annotation_count"] for x in rows]
    selected = []
    for q in (0.25, 0.50, 0.75):
        target = quantile(values, q)
        row = min(rows, key=lambda x: (abs(x["annotation_count"] - target), x["scene"]))
        row = dict(row)
        row["percentile"] = q
        row["target_annotation_count"] = target
        selected.append(row)

    selected_ids = {x["video_id"] for x in selected}
    subset = dict(data)
    subset["videos"] = [x for x in data["videos"] if x["id"] in selected_ids]
    subset["images"] = [x for x in data["images"] if x["video_id"] in selected_ids]
    image_ids = {x["id"] for x in subset["images"]}
    subset["annotations"] = [x for x in data["annotations"] if x["image_id"] in image_ids]

    args.subset_json.parent.mkdir(parents=True, exist_ok=True)
    args.subset_json.write_text(json.dumps(subset, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "method": "nearest annotation-count scene to linear percentile; ties by scene name",
        "percentiles": [0.25, 0.50, 0.75],
        "source_json": str(args.input),
        "source_sha256": sha256(args.input),
        "selected": selected,
        "selected_video_ids": sorted(selected_ids),
        "subset_json": str(args.subset_json),
        "subset_sha256": sha256(args.subset_json),
        "subset_image_count": len(subset["images"]),
        "subset_annotation_count": len(subset["annotations"]),
        "gt_objects_unchanged": True,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
