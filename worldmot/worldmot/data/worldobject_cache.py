"""Immutable V1/V2 support reader with GT-free observations and exact ROI join."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from .worldobject_feature_join import FrozenGMTFeatureJoin
from ..models.world_object_observation import normalized_feature, FeatureUnavailable

V1 = "worldobject_omega512_sparse_support_v1"
V2 = "worldobject_omega512_sparse_support_v2"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


class WorldObjectCache:
    def __init__(self, root, scene, *, raw_cache=None, require_features=True, require_geometry=False):
        self.root, self.scene = Path(root), int(scene)
        self.path = self.root / f"scene_{self.scene:02d}"
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if self.manifest.get("status") != "COMPLETE":
            raise RuntimeError("INCOMPLETE_SCENE_CACHE")
        schema = self.manifest.get("schema")
        if schema not in (V1, V2):
            raise ValueError("UNSUPPORTED_WORLDOBJECT_CACHE_SCHEMA")
        if require_geometry and schema != V2:
            raise ValueError("V1_CAMERA_DEPTH_SUPPORT_REQUIRES_REBUILD: no lossless geometry migration")
        self.require_features = require_features
        self.require_geometry = require_geometry
        if require_geometry and self.manifest.get("support_contract")!="camera_forward_depth_to_persistent_world_v2":
            raise ValueError("V2_CAMERA_DEPTH_CONTRACT_REQUIRED")
        self.joiner = FrozenGMTFeatureJoin(raw_cache, self.manifest) if raw_cache is not None else None
        if schema == V1 and require_features and self.joiner is None:
            raise FeatureUnavailable("FEATURE_UNAVAILABLE: V1 requires read-only raw GMT feature join")
        self.feature_dim = self.joiner.feature_dim if self.joiner else self.manifest.get("feature_dim")
        self.schema = schema

    def frames(self):
        previous = None
        count = 0
        for entry in self.manifest["shards"]:
            path = self.path / entry["file"]
            if path.parent != self.path or sha256(path) != entry["sha256"]:
                raise RuntimeError("SUPPORT_SHARD_SHA_OR_PATH_MISMATCH")
            sidepath = path.with_suffix(".json")
            if self.schema == V2 and sha256(sidepath) != entry.get("sidecar_sha256"):
                raise RuntimeError("SUPPORT_SIDECAR_SHA_MISMATCH")
            side = json.loads(sidepath.read_text())
            with np.load(path, allow_pickle=False) as payload:
                arrays = dict(payload)
            timestamps = arrays["timestamps"].astype(int).tolist()
            for i, timestamp in enumerate(timestamps):
                if previous is not None and timestamp <= previous:
                    raise ValueError("NONCAUSAL_SUPPORT_CACHE_ORDER")
                previous, count = timestamp, count + 1
                lo, hi = map(int, arrays["frame_offsets"][i:i+2])
                original_record = side["records"][i]
                # Deliberate allowlist: legacy identity arrays and accidental
                # target fields are never exposed by the inference adapter.
                record = {k: original_record[k] for k in (
                    "frame_id", "alignment", "original_view_alignment", "camera_ids", "image_ids", "rgb_sha256",
                    "raw_candidate_shard_sha256", "runtime", "camera_models") if k in original_record}
                rows = []
                for j in range(lo, hi):
                    bc, fc = int(arrays["box_support_count"][j]), int(arrays["foreground_support_count"][j])
                    row = {"view": int(arrays["camera_ids"][j]), "detection_id": int(arrays["detection_ids"][j]),
                           "bbox": arrays["bbox"][j].astype(float), "score": float(arrays["scores"][j]),
                           "center": arrays["center"][j].astype(float), "cov": arrays["covariance"][j].astype(float),
                           "box": arrays["box_support"][j,:bc].astype(float),
                           "foreground": arrays["foreground_support"][j,:fc].astype(float),
                           "quality": float(arrays["quality"][j]),
                           "foreground_fraction": float(arrays["foreground_fraction"][j]),
                           "ambiguity_code": int(arrays["ambiguity"][j]),
                           "geometry_valid": bool(self.schema == V2 and arrays["geometry_valid"][j]),
                           "geometry_status": ("LEGACY_CAMERA_DEPTH_UNVERIFIED" if self.schema == V1
                                               else str(arrays["geometry_status"][j])),
                           "coordinate_frame": "PERSISTENT_WORLD"}
                    if self.schema == V2:
                        row["roi_feature"] = arrays["roi_features"][j].copy()
                        row["feature_valid"] = bool(arrays["feature_valid"][j])
                        if not record.get("alignment",{}).get("success",False):
                            row["geometry_valid"]=False
                            row["geometry_status"]="WORLD_ALIGNMENT_FAILED"
                    rows.append(row)
                if self.joiner is not None:
                    rows = self.joiner.join(self.scene, timestamp, rows, record)
                for row in rows:
                    if self.require_features:
                        if not row.get("feature_valid",False):
                            raise FeatureUnavailable("FEATURE_UNAVAILABLE: cache declares invalid ROI descriptor")
                        row["roi_feature"] = normalized_feature(row.get("roi_feature"), self.feature_dim)
                        row["feature_valid"] = True
                keys = [(r["view"], r["detection_id"]) for r in rows]
                if len(keys) != len(set(keys)):
                    raise ValueError("DUPLICATE_SUPPORT_CANDIDATE_KEY")
                yield {"timestamp": timestamp, "rows": rows, "record": record, "scene": self.scene,
                       "source": {"schema": self.schema, "support_shard_sha256": entry["sha256"],
                                  "source_commit": self.manifest.get("source_commit")}}
        if count != self.manifest["timestamps"]:
            raise RuntimeError("SUPPORT_TIMESTAMP_COVERAGE_MISMATCH")


def iter_complete_scenes(root, scenes=None, **kwargs):
    root = Path(root)
    wanted = None if scenes is None else set(map(int, scenes))
    for path in sorted(root.glob("scene_*")):
        if not (path / "manifest.json").is_file():
            continue
        sid = int(path.name.split("_")[1])
        if wanted is None or sid in wanted:
            yield WorldObjectCache(root, sid, **kwargs)
