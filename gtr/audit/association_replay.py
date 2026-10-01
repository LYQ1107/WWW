"""CPU observation cache and deterministic association replay helpers.

The cache is deliberately placed after ``GTRRCNN.inference`` and before any
track state is updated.  It therefore contains the released detector/ROI
observation stream (boxes, scores, classes and fused re-identification
features) without pixels, GT, or track IDs.  Replay reconstructs Detectron2
``Instances`` on the model device and lets the unchanged GMT association code
consume those observations.

The module is inert unless ``GMT_ASSOC_REPLAY_DUMP`` or
``GMT_ASSOC_REPLAY_LOAD`` is explicitly set.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from detectron2.structures import Boxes, Instances


def scene_name(batch_item: dict[str, Any]) -> str:
    """Return the canonical scene token used by the audit JSON."""
    name = str(batch_item.get("file_name", "scene"))
    for part in Path(name).parts:
        if re.search(r"_View\d+$", part):
            return re.sub(r"_View\d+$", "", part)
    match = re.search(r"([^/\\]+)_View\d+", name)
    return match.group(1) if match else Path(name).stem


def _cpu(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return value


def observation_record(inst, item: dict[str, Any], *, frame_index: int, view_index: int) -> dict[str, Any]:
    """Serialize one post-ROI observation without mutable tracker state."""
    if not inst.has("pred_boxes") or not inst.has("reid_features"):
        raise ValueError("GMT replay observations require pred_boxes and reid_features")
    boxes = inst.pred_boxes.tensor.detach().to("cpu", dtype=torch.float32).contiguous()
    scores = inst.scores.detach().to("cpu", dtype=torch.float32).contiguous()
    classes = inst.pred_classes.detach().to("cpu", dtype=torch.int64).contiguous()
    reid = inst.reid_features.detach().to("cpu", dtype=torch.float32).contiguous()
    return {
        "frame_index": int(frame_index),
        "view_index": int(view_index),
        "image_id": int(item.get("image_id", -1)),
        "video_id": int(item.get("video_id", -1)),
        "frame_id": int(item.get("frame_id", frame_index)),
        "view_id": int(item.get("view_id", view_index + 1)),
        "file_name": str(item.get("file_name", "")),
        "width": int(item.get("width", inst.image_size[1])),
        "height": int(item.get("height", inst.image_size[0])),
        "image_size": tuple(int(x) for x in inst.image_size),
        "pred_boxes": boxes,
        "scores": scores,
        "pred_classes": classes,
        "reid_features": reid,
    }


class AssociationReplayWriter:
    """Collect one scene and atomically write ``<scene>.pt``."""

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, list[dict[str, Any]]] = {}

    def append(self, scene: str, record: dict[str, Any]) -> None:
        self._records.setdefault(str(scene), []).append(record)

    def write_scene(self, scene: str) -> Path:
        rows = self._records.get(str(scene), [])
        rows = sorted(rows, key=lambda x: (x["frame_index"], x["view_index"]))
        payload = {
            "format": "gmt-association-replay-v1",
            "scene": str(scene),
            "records": rows,
            "record_count": len(rows),
            "dtype_policy": "float32_cpu",
        }
        path = self.root / f"{scene}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        meta = path.with_suffix(".json")
        meta.write_text(json.dumps({
            "format": payload["format"],
            "scene": str(scene),
            "record_count": len(rows),
            "first_frame": rows[0]["frame_index"] if rows else None,
            "last_frame": rows[-1]["frame_index"] if rows else None,
            "views": sorted({int(x["view_index"]) for x in rows}),
        }, indent=2) + "\n")
        return path


class AssociationReplayReader:
    """Load and index a frozen scene observation cache."""

    def __init__(self, root: str | os.PathLike[str], scene: str):
        path = Path(root) / f"{scene}.pt"
        if not path.exists():
            raise FileNotFoundError(f"association replay cache missing: {path}")
        # The cache is generated locally by this audit; no pickle data from an
        # external source is accepted by this reader.
        payload = torch.load(path, map_location="cpu")
        if payload.get("format") != "gmt-association-replay-v1":
            raise ValueError(f"unsupported replay cache format in {path}")
        self.scene = str(payload["scene"])
        self.records = {
            (int(row["frame_index"]), int(row["view_index"])): row
            for row in payload["records"]
        }

    def get(self, frame_index: int, view_index: int, device) -> Instances:
        key = (int(frame_index), int(view_index))
        if key not in self.records:
            raise KeyError(f"replay cache has no {self.scene} frame/view {key}")
        row = self.records[key]
        inst = Instances(tuple(int(x) for x in row["image_size"]))
        inst.pred_boxes = Boxes(row["pred_boxes"].to(device=device, dtype=torch.float32))
        inst.scores = row["scores"].to(device=device, dtype=torch.float32)
        inst.pred_classes = row["pred_classes"].to(device=device, dtype=torch.int64)
        inst.reid_features = row["reid_features"].to(device=device, dtype=torch.float32)
        return inst


def write_tracking_trace(root: str | os.PathLike[str], scene: str, instances) -> Path:
    """Write audit-only post-association IDs for exact baseline/replay checks."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, inst in enumerate(instances):
        ids = inst.track_ids.detach().to("cpu", dtype=torch.int64).contiguous()
        rows.append({
            "sequence_index": int(index),
            "track_ids": ids,
            "count": int(len(inst)),
            "pred_boxes": (inst.pred_boxes.tensor.detach().to("cpu", dtype=torch.float32).contiguous()
                            if inst.has("pred_boxes") else None),
            "scores": (inst.scores.detach().to("cpu", dtype=torch.float32).contiguous()
                        if inst.has("scores") else None),
            "pred_classes": (inst.pred_classes.detach().to("cpu", dtype=torch.int64).contiguous()
                              if inst.has("pred_classes") else None),
        })
    path = root / f"{scene}.pt"
    tmp = path.with_suffix(".pt.tmp")
    torch.save({
        "format": "gmt-association-trace-v1",
        "scene": str(scene),
        "records": rows,
    }, tmp)
    os.replace(tmp, path)
    return path
