"""Read-only exact join of sparse Ω support with the frozen pre-GTA GMT cache."""
from __future__ import annotations
import numpy as np
from ..models.world_object_observation import normalized_feature


class FrozenGMTFeatureJoin:
    def __init__(self, raw_cache, sparse_manifest):
        self.raw_cache = raw_cache
        self.feature_dim = raw_cache.feature_dim
        expected = sparse_manifest.get("detector_cache", {}).get("cache_lock_sha256")
        actual = raw_cache.manifest()["cache_lock_sha256"]
        if not expected or expected != actual:
            raise ValueError("DETECTOR_LOCK_SHA_MISMATCH_OR_MISSING")

    def join(self, scene, timestamp, rows, record):
        if not record.get("image_ids"):
            raise ValueError("JOIN_IMAGE_IDS_REQUIRED")
        detections = self.raw_cache.get(scene, timestamp, image_ids=record["image_ids"])
        raw = {(int(view), d.detection_id): d for view, ds in detections.items() for d in ds}
        keys = [(int(r["view"]), int(r["detection_id"])) for r in rows]
        if len(keys) != len(set(keys)) or set(keys) != set(raw):
            raise ValueError("JOIN_CANDIDATE_COUNT_OR_ONE_TO_ONE_MISMATCH")
        if record.get("raw_candidate_shard_sha256") and record["raw_candidate_shard_sha256"] != self.raw_cache.last_diagnostics["raw_candidate_shard_sha256"]:
            raise ValueError("JOIN_CANDIDATE_SHARD_SHA_MISMATCH")
        if record.get("rgb_sha256") and record["rgb_sha256"] != self.raw_cache.last_diagnostics["rgb_sha256"]:
            raise ValueError("JOIN_RGB_SHA_MISMATCH")
        result = []
        for row, key in zip(rows, keys):
            det = raw[key]
            if not np.allclose(row["bbox"], det.bbox_xywh, atol=1e-5, rtol=1e-6):
                raise ValueError("JOIN_BBOX_MISMATCH")
            if not np.isclose(row["score"], det.score, atol=1e-6, rtol=1e-6):
                raise ValueError("JOIN_SCORE_MISMATCH")
            f = normalized_feature(det.detector_roi_feature, self.feature_dim)
            if "roi_feature" in row and not np.allclose(f, normalized_feature(row["roi_feature"], self.feature_dim), atol=1e-6):
                raise ValueError("JOIN_EMBEDDED_FEATURE_MISMATCH")
            result.append({**row, "roi_feature": f, "feature_valid": True,
                           "feature_source": "FROZEN_PRE_GTA_GMT_ROI_READONLY_JOIN"})
        return result
