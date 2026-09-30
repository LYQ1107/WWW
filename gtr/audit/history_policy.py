"""Audit-only history weighting policies.

No GT information is used here.  The policy is called only when the
GMT_AUDIT_HISTORY_POLICY environment variable is present.
"""
from __future__ import annotations

import math
from typing import Optional

import torch


def _field_values(instances, field: str) -> list[torch.Tensor]:
    values = []
    for inst in instances:
        if inst.has(field):
            values.append(getattr(inst, field).detach().float())
        else:
            values.append(torch.ones((len(inst),), device=inst.track_ids.device if inst.has("track_ids") else "cpu"))
    return values


def _quality_values(instances, policy: str) -> torch.Tensor:
    if policy == "score":
        for field in ("scores", "objectness_logits"):
            if all(inst.has(field) for inst in instances):
                return torch.cat(_field_values(instances, field), dim=0)
        return torch.cat(_field_values(instances, "scores"), dim=0)
    if policy == "area":
        out = []
        for inst in instances:
            boxes = inst.pred_boxes if inst.has("pred_boxes") else inst.proposal_boxes
            box = boxes.tensor.detach().float()
            area = (box[:, 2] - box[:, 0]).clamp(min=0) * (box[:, 3] - box[:, 1]).clamp(min=0)
            height, width = inst.image_size
            out.append(area / max(float(height * width), 1.0))
        return torch.cat(out, dim=0) if out else torch.empty(0)
    raise ValueError("unsupported audit history policy: %s" % policy)


def build_history_weights(
    instances,
    k: Optional[int],
    ids: torch.Tensor,
    policy: str,
    keep_ratio: float,
) -> torch.Tensor:
    """Return a 0/1 weight for each flattened historical observation.

    ``instances`` must be in the same order used to flatten ``ids``.  If a
    complete window is supplied, ``k`` identifies the current instance and is
    excluded; audit callers normally pass the already-filtered history and
    ``k=None``.  For each existing ID, the top ``ceil(keep_ratio*n)`` quality
    observations are retained.  Ties are resolved by the original flattened
    index, so the operation is deterministic.
    """
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    if k is not None:
        instances = [x for i, x in enumerate(instances) if i != k]
    values = _quality_values(instances, policy).to(device=ids.device)
    if values.numel() != ids.numel():
        raise ValueError("history quality count does not match ids")
    weights = torch.zeros((ids.numel(),), dtype=torch.float32, device=ids.device)
    for uid in torch.unique(ids).tolist():
        positions = torch.nonzero(ids == uid, as_tuple=False).flatten().tolist()
        keep = max(1, int(math.ceil(len(positions) * keep_ratio)))
        positions.sort(key=lambda p: (-float(values[p].item()), p))
        weights[torch.as_tensor(positions[:keep], device=ids.device)] = 1.0
    return weights
