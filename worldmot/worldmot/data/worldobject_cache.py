"""Checksum-checked reader for the WorldObject sparse support cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


class WorldObjectCache:
    def __init__(self, root: str | Path, scene: int):
        self.root = Path(root); self.scene = int(scene); self.path = self.root / f"scene_{self.scene:02d}"
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if self.manifest.get("status") != "COMPLETE":
            raise RuntimeError(f"scene cache is not complete: {self.path}")
        if self.manifest.get("schema") != "worldobject_omega512_sparse_support_v1":
            raise RuntimeError(f"unexpected cache schema: {self.manifest.get('schema')}")

    def frames(self):
        for entry in self.manifest["shards"]:
            path = self.path / entry["file"]
            if sha256(path) != entry["sha256"]:
                raise RuntimeError(f"checksum mismatch: {path}")
            side = json.loads(path.with_suffix(".json").read_text())
            arrays = dict(np.load(path, allow_pickle=False))
            timestamps = arrays["timestamps"].astype(int).tolist()
            for i, timestamp in enumerate(timestamps):
                lo, hi = [int(x) for x in arrays["frame_offsets"][i:i + 2]]
                rows = []
                for j in range(lo, hi):
                    bcount = int(arrays["box_support_count"][j]); fcount = int(arrays["foreground_support_count"][j])
                    ambiguity_code = int(arrays["ambiguity"][j]) if "ambiguity" in arrays else -1
                    rows.append({"view": int(arrays["camera_ids"][j]), "detection_id": int(arrays["detection_ids"][j]),
                                 "bbox": arrays["bbox"][j].astype(float), "score": float(arrays["scores"][j]),
                                 "center": arrays["center"][j].astype(float), "cov": arrays["covariance"][j].astype(float),
                                 "box": arrays["box_support"][j, :bcount].astype(float),
                                 "foreground": arrays["foreground_support"][j, :fcount].astype(float),
                                 "quality": float(arrays["quality"][j]), "foreground_fraction": float(arrays["foreground_fraction"][j]),
                                 "ambiguity_code": ambiguity_code,
                                 "identity": int(arrays["identity"][j])})
                record = side.get("records", [])[i] if i < len(side.get("records", [])) else {"frame_id": int(timestamp)}
                yield {"timestamp": int(timestamp), "rows": rows, "record": record,
                       "scene": self.scene, "shard": entry["file"]}


def iter_complete_scenes(root: str | Path, scenes=None):
    root = Path(root); wanted = [int(x) for x in scenes] if scenes is not None else None
    for path in sorted(root.glob("scene_*")):
        if not path.is_dir() or not (path / "manifest.json").is_file(): continue
        sid = int(path.name.split("_")[1])
        if wanted is not None and sid not in wanted: continue
        yield WorldObjectCache(root, sid)
