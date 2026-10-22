"""PWQT losses with explicit null, visibility, birth and world consistency terms."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class WorldQueryCriterion(nn.Module):
    def __init__(self, identity_weight: float = 1.0, assignment_weight: float = 1.0,
                 birth_weight: float = 0.5, visibility_weight: float = 0.5,
                 projection_weight: float = 0.25, consistency_weight: float = 0.25):
        super().__init__()
        self.weights = {"identity": identity_weight, "assignment": assignment_weight,
                        "birth": birth_weight, "visibility": visibility_weight,
                        "projection": projection_weight, "consistency": consistency_weight,
                        "existence": 0.5}

    def forward(self, output: dict, targets: dict) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        logits = output["assignment_logits"]
        assignment = targets.get("assignment")
        if assignment is not None:
            assignment = assignment.to(logits.device)
            if assignment.shape == logits.shape:
                # Multi-view supervision is multi-label: one query may have
                # one positive column in each camera plus a null column when
                # it has no current observation.
                target = assignment.float()
                weight = torch.ones_like(target)
                valid = targets.get("observation_valid")
                if valid is not None:
                    valid = valid.to(logits.device).bool()
                    weight[..., :-1] = valid[:, None, :].float()
                positive = target[..., :-1].sum().detach()
                negative = weight[..., :-1].sum() - positive
                pos_weight = torch.ones_like(target)
                pos_weight[..., :-1] = (negative / positive.clamp_min(1.0)).clamp_min(1.0).clamp_max(64.0)
                losses["assignment"] = F.binary_cross_entropy_with_logits(
                    logits, target, weight=weight, pos_weight=pos_weight, reduction="sum"
                ) / weight.sum().clamp_min(1.0)
            else:
                losses["assignment"] = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), assignment.reshape(-1), ignore_index=-100
                )
            existence_target = targets.get("existence")
            if existence_target is None:
                if assignment.shape == logits.shape:
                    existence_target = (assignment[..., :-1].sum(dim=-1) > 0).float()
                else:
                    existence_target = (assignment != (logits.shape[-1] - 1)).float()
            existence_target = existence_target.to(output["existence_logits"].device).float()
            positive = existence_target.sum().detach()
            negative = existence_target.numel() - positive
            pos_weight = (negative / positive.clamp_min(1.0)).clamp_min(1.0).clamp_max(64.0)
            losses["existence"] = F.binary_cross_entropy_with_logits(
                output["existence_logits"], existence_target,
                pos_weight=pos_weight,
            )
        birth_target = targets.get("birth")
        if birth_target is not None:
            birth_target = birth_target.to(output["birth_logits"].device).float()
            weight = torch.ones_like(birth_target)
            valid = targets.get("observation_valid")
            if valid is not None:
                weight = valid.to(weight.device).float()
            losses["birth"] = F.binary_cross_entropy_with_logits(
                output["birth_logits"], birth_target, weight=weight, reduction="sum"
            ) / weight.sum().clamp_min(1.0)
        visible_target = targets.get("visibility")
        if visible_target is not None:
            losses["visibility"] = F.binary_cross_entropy_with_logits(output["visibility_logits"], visible_target.float())
        projection_target = targets.get("projection")
        if projection_target is not None:
            valid = targets.get("projection_valid", torch.ones_like(projection_target[..., 0], dtype=torch.bool))
            losses["projection"] = F.smooth_l1_loss(output["projection_residual"][valid], projection_target[valid]) if valid.any() else output["projection_residual"].sum() * 0.0
        current_xyz = targets.get("xyz")
        if current_xyz is not None:
            valid = torch.isfinite(current_xyz).all(-1)
            losses["consistency"] = F.smooth_l1_loss(output["xyz"][valid], current_xyz[valid]) if valid.any() else output["xyz"].sum() * 0.0
        total = output["assignment_logits"].sum() * 0.0
        for name, value in losses.items():
            total = total + float(self.weights.get(name, 1.0)) * value
        losses["total"] = total
        return losses
