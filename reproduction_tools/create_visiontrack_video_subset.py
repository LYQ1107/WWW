"""Create a COCO/ VisionTrack annotation subset for one video.

The output preserves the original video IDs and dataset metadata, while
retaining only images and annotations belonging to ``--video-id``.  It is
intended for bounded native/runtime diagnostics; it never edits the source
annotation file in place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--video-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.input.read_text(encoding="utf-8"))
    images = [
        item for item in source.get("images", [])
        if int(item.get("video_id", -1)) == int(args.video_id)
    ]
    image_ids = {int(item["id"]) for item in images}
    if not images:
        raise ValueError(f"video {args.video_id} has no images in {args.input}")
    annotations = [
        item for item in source.get("annotations", [])
        if int(item.get("image_id", -1)) in image_ids
    ]
    subset: dict[str, Any] = {
        key: source[key]
        for key in ("info", "licenses", "categories", "videos")
        if key in source
    }
    subset.update({"images": images, "annotations": annotations})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(subset, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "source": str(args.input.resolve()),
        "source_sha256": sha256(args.input.resolve()),
        "output": str(args.output.resolve()),
        "video_id": int(args.video_id),
        "image_count": len(images),
        "annotation_count": len(annotations),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
