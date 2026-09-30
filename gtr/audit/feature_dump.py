"""Binary, audit-only feature dump for entity retrieval diagnostics."""
from __future__ import annotations

import re
from pathlib import Path

import torch


def _scene_name(batch_item: dict) -> str:
    name = str(batch_item.get("file_name", "scene"))
    # Dataset mappers make file_name absolute.  Select the path component
    # carrying the canonical ``<scene>_ViewN`` token rather than Path.parts[0]
    # (which is ``/`` for an absolute path).
    for part in Path(name).parts:
        if re.search(r"_View\d+$", part):
            return re.sub(r"_View\d+$", "", part)
    match = re.search(r"([^/\\]+)_View\d+", name)
    return match.group(1) if match else Path(name).stem


def dump_scene_observations(instances, batch, audit_dir) -> Path:
    """Write one ``.pt`` file for the current video after tracking.

    Raw appearance and fused features are stored as float16 CPU tensors.  No
    image pixels are copied and this function is never called unless the
    caller explicitly sets ``GMT_AUDIT_DUMP_FEATURES=1``.
    """
    root = Path(audit_dir) / "features"
    root.mkdir(parents=True, exist_ok=True)
    if len(instances) != len(batch):
        raise ValueError("instance/batch length mismatch")
    records = []
    for inst, item in zip(instances, batch):
        if inst.has("pred_boxes"):
            boxes = inst.pred_boxes.tensor.detach().cpu().float()
        elif inst.has("proposal_boxes"):
            boxes = inst.proposal_boxes.tensor.detach().cpu().float()
        else:
            raise ValueError("audit feature dump needs boxes")
        scores = inst.scores.detach().cpu().float() if inst.has("scores") else torch.ones(len(inst))
        ids = inst.track_ids.detach().cpu().long() if inst.has("track_ids") else torch.full((len(inst),), -1)
        fused = inst.reid_features.detach().cpu().half() if inst.has("reid_features") else None
        appearance = inst.audit_app_features.detach().cpu().half() if inst.has("audit_app_features") else None
        if appearance is None or fused is None:
            raise ValueError("audit feature fields are missing; enable GMT_AUDIT_DUMP_FEATURES")
        for j in range(len(inst)):
            records.append({
                "image_id": int(item.get("image_id", -1)),
                "frame_id": int(item.get("frame_id", -1)),
                "view_id": int(item.get("view_id", -1)),
                "video_id": int(item.get("video_id", -1)),
                "width": int(item.get("width", inst.image_size[1])),
                "height": int(item.get("height", inst.image_size[0])),
                "pred_track_id": int(ids[j]),
                "bbox_xyxy": boxes[j],
                "score": float(scores[j]),
                "appearance_feature": appearance[j],
                "fused_feature": fused[j],
            })
    scene = _scene_name(batch[0]) if batch else "scene"
    path = root / (scene + ".pt")
    torch.save({"scene": scene, "records": records}, path)
    return path
