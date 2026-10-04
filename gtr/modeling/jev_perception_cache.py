"""Frozen detector/ReID cache used by the reviewer-proof v2 rollouts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Optional, Tuple

import torch


CACHE_VERSION = "frozen_perception_cache_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _tensor(value: Any, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if value is None:
        return torch.empty((0,), dtype=dtype)
    if hasattr(value, "detach"):
        return value.detach().cpu().to(dtype=dtype).contiguous()
    return torch.as_tensor(value, dtype=dtype).cpu().contiguous()


def _field(instances: Any, name: str, default: Any = None) -> Any:
    try:
        return instances.get(name)
    except (AttributeError, KeyError, TypeError):
        try:
            return getattr(instances, name)
        except AttributeError:
            return default


class FrozenPerceptionCacheWriter:
    """Write one immutable tensor payload per frame/view and an index JSONL."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.records = self.root / "records"
        self.records.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.jsonl"
        self.handle = self.index_path.open("a", encoding="utf-8")
        self._seen = set()
        if self.index_path.stat().st_size:
            with self.index_path.open(encoding="utf-8") as existing:
                for line in existing:
                    if line.strip():
                        payload = json.loads(line)
                        self._seen.add(self.key(payload))

    @staticmethod
    def key(payload: Mapping[str, Any]) -> Tuple[int, int, int]:
        return (
            int(payload["video_id"]),
            int(payload["frame"]),
            int(payload["view"]),
        )

    def write(
        self,
        *,
        video_id: int,
        frame: int,
        view: int,
        instances: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        payload_index = {
            "cache_version": CACHE_VERSION,
            "video_id": int(video_id),
            "frame": int(frame),
            "view": int(view),
        }
        key = self.key(payload_index)
        if key in self._seen:
            raise RuntimeError(f"refusing to overwrite cached perception record {key}")

        boxes = _tensor(getattr(instances, "pred_boxes").tensor)
        scores_value = _field(instances, "scores")
        score_field = "scores"
        if scores_value is None:
            scores_value = _field(instances, "objectness_logits")
            score_field = "objectness_logits"
        scores = _tensor(scores_value).reshape(-1)
        reid_value = _field(instances, "reid_features")
        reid_features = _tensor(reid_value)
        if reid_features.numel() == 0:
            reid_features = torch.empty((len(boxes), 0), dtype=torch.float32)
        if len(scores) != len(boxes) or reid_features.shape[0] != len(boxes):
            raise ValueError(
                "perception fields disagree on detection count: "
                f"boxes={len(boxes)} scores={len(scores)} reid={reid_features.shape[0]}"
            )

        proposal_metadata: Dict[str, Any] = {
            "score_field": score_field,
            "has_reid_features": bool(reid_features.shape[1] > 0),
        }
        for name in ("pred_classes", "objectness_logits", "scores"):
            value = _field(instances, name)
            if value is not None:
                proposal_metadata[name] = _tensor(value).reshape(-1).tolist()
        image_size = tuple(int(value) for value in getattr(instances, "image_size", (0, 0)))
        file_name = f"video_{int(video_id):08d}_frame_{int(frame):08d}_view_{int(view):04d}.pt"
        target = self.records / file_name
        fd, temporary_name = tempfile.mkstemp(prefix=f".{file_name}.", dir=str(self.records))
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            torch.save(
                {
                    "cache_version": CACHE_VERSION,
                    "video_id": int(video_id),
                    "frame": int(frame),
                    "view": int(view),
                    "pred_boxes": boxes,
                    "detection_scores": scores,
                    "reid_features": reid_features,
                    "image_size": image_size,
                    "proposal_metadata": proposal_metadata,
                    "metadata": dict(metadata or {}),
                },
                str(temporary),
            )
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()

        record = {
            **payload_index,
            "record": str(target.relative_to(self.root)),
            "record_sha256": _sha256(target),
            "detection_count": int(len(boxes)),
            "reid_dim": int(reid_features.shape[1]),
            "image_size": list(image_size),
            "proposal_metadata": proposal_metadata,
            "metadata": dict(metadata or {}),
        }
        self.handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.handle.flush()
        self._seen.add(key)
        return record

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class FrozenPerceptionCache:
    """Read-only indexed view over a cache produced by the writer."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.index_path = self.root / "index.jsonl"
        if not self.index_path.is_file():
            raise FileNotFoundError(self.index_path)
        self.index: Dict[Tuple[int, int, int], Mapping[str, Any]] = {}
        with self.index_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                key = FrozenPerceptionCacheWriter.key(payload)
                if key in self.index:
                    raise ValueError(f"duplicate cache key at line {line_number}: {key}")
                self.index[key] = payload

    def keys(self):
        return tuple(sorted(self.index))

    def load(self, video_id: int, frame: int, view: int) -> Mapping[str, Any]:
        key = (int(video_id), int(frame), int(view))
        if key not in self.index:
            raise KeyError(key)
        metadata = self.index[key]
        path = self.root / str(metadata["record"])
        if _sha256(path) != metadata["record_sha256"]:
            raise ValueError(f"perception cache checksum mismatch: {path}")
        payload = torch.load(str(path), map_location="cpu")
        if payload.get("cache_version") != CACHE_VERSION:
            raise ValueError(f"unsupported cache version in {path}")
        return payload
