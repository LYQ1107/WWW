#!/usr/bin/env python3
"""Freeze the scene-level decision-formulation split before results exist."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scene_name(file_name: str) -> str:
    match = re.search(r"([^/\\]+)_View\d+", file_name)
    if not match:
        raise ValueError(f"cannot derive scene from {file_name!r}")
    return match.group(1)


def main() -> None:
    annotation = Path("datasets/VisionTrack/annotations/train.json")
    payload = json.loads(annotation.read_text())
    scenes = sorted({scene_name(str(row["file_name"])) for row in payload["images"]})
    ordered = sorted(scenes, key=lambda s: (hashlib.sha256(s.encode()).hexdigest(), s))
    n = len(ordered)
    ideal = [0.70 * n, 0.10 * n, 0.20 * n]
    counts = [math.floor(x) for x in ideal]
    for index in sorted(range(3), key=lambda i: (ideal[i] - counts[i], -i), reverse=True)[: n - sum(counts)]:
        counts[index] += 1
    counts[0] = max(1, counts[0])
    counts[1] = max(1, counts[1])
    counts[2] = n - counts[0] - counts[1]
    if counts[2] < 1:
        raise RuntimeError(f"invalid split counts {counts}")
    a, b = counts[0], counts[0] + counts[1]
    groups = {
        "JEV-train": ordered[:a],
        "JEV-calibration": ordered[a:b],
        "JEV-dev": ordered[b:],
    }
    manifest = {
        "format": "gmt-decision-formulation-scene-split-v1",
        "source_annotation": str(annotation.resolve()),
        "source_annotation_sha256": sha256(annotation),
        "ordering": "SHA256(scene UTF-8 bytes), then scene name",
        "ratios_requested": {"JEV-train": 0.70, "JEV-calibration": 0.10, "JEV-dev": 0.20},
        "rounding": "largest remainder; every split has at least one scene",
        "scene_count": n,
        "counts": {key: len(value) for key, value in groups.items()},
        "scenes_by_split": groups,
        "scene_hashes": {scene: hashlib.sha256(scene.encode()).hexdigest() for scene in ordered},
        "frozen_before_records": True,
        "visiontrack_test_used": False,
    }
    out = Path("decision_audit/manifests/JEV_DECISION_SPLIT.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(out), "counts": manifest["counts"], "scenes": groups}, indent=2))


if __name__ == "__main__":
    main()
