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

class TemporalQueryCriterion(nn.Module):
    """Balanced active-query/NULL CE and newborn/multi-view supervision.

    Duplicate GT targets (-2) and padding never become negative identities.
    Invisible active tracks remain alive for their registered miss window.
    """
    def forward(self,output,identity,valid,cameras,previous_ids,step_mask):
        logits=output['assignment_logits'];b,q,total=logits.shape;n=total-1
        observed_valid = valid & step_mask[:, None]
        valid=observed_valid & (identity!=-2)
        active=previous_ids>=0
        scores=(logits[...,:-1]-logits[...,-1:]).transpose(1,2).masked_fill(~active[:,None,:],-1e4)
        classes=torch.full((b,n),q,device=logits.device,dtype=torch.long)
        same=(identity[:,:,None]==previous_ids[:,None,:]) & active[:,None,:] & (identity[:,:,None]>=0)
        seen=same.any(-1);classes=torch.where(seen,same.float().argmax(-1),classes)
        ce=F.cross_entropy(torch.cat([scores,scores.new_zeros((b,n,1))],-1).reshape(-1,q+1),classes.reshape(-1),reduction='none').reshape(b,n)
        # Average positives and NULLs independently to avoid majority collapse.
        pos=valid&seen;neg=valid&~seen
        assignment=(ce*pos).sum()/pos.sum().clamp_min(1)+(ce*neg).sum()/neg.sum().clamp_min(1)
        newborn=(identity>=0)&~seen
        birth_weights=valid.float();positive=(birth_weights*newborn).sum();negative=(birth_weights*(~newborn)).sum()
        # Birth is rare over a continuous clip, even when one frame contains
        # many newborns.  Per-frame ratios alone collapse to all-negative.
        pw=(negative/positive.clamp_min(1)).clamp(10,30)
        birth=F.binary_cross_entropy_with_logits(output['birth_logits'],newborn.float(),pos_weight=pw,reduction='none')
        birth=(birth*birth_weights).sum()/birth_weights.sum().clamp_min(1)
        pair_mask=valid[:,:,None]&valid[:,None,:]&(cameras[:,:,None]!=cameras[:,None,:])
        pair_target=(identity[:,:,None]==identity[:,None,:])&(identity[:,:,None]>=0)
        ppos=(pair_target&pair_mask).sum();pneg=(~pair_target&pair_mask).sum();ppw=(pneg/ppos.clamp_min(1)).clamp(1,30)
        pair=F.binary_cross_entropy_with_logits(output['pair_logits'],pair_target.float(),pos_weight=ppw,reduction='none')
        pair=(pair*pair_mask).sum()/pair_mask.sum().clamp_min(1)
        visible=(same&valid[:,:,None]).any(1)
        # Active but temporarily occluded queries are positive existence.
        existence_target=active.float();emask=step_mask[:,None].expand(-1,q)
        epw=((~active & emask).sum()/(active & emask).sum().clamp_min(1)).clamp(1,32)
        existence=F.binary_cross_entropy_with_logits(output['existence_logits'],existence_target,pos_weight=epw,reduction='none')
        existence=(existence*emask).sum()/emask.sum().clamp_min(1)
        visibility=output['visibility_logits'].new_zeros(output['visibility_logits'].shape)
        for c in range(visibility.shape[-1]):visibility[:,:,c]=(same & valid[:,:,None] & (cameras[:,:,None]==c+1)).any(1)
        vloss=F.binary_cross_entropy_with_logits(output['visibility_logits'],visibility,reduction='none')
        vloss=(vloss*emask[:,:,None]).sum()/(emask.sum()*visibility.shape[-1]).clamp_min(1)
        total_loss=assignment+birth+pair+0.25*existence+0.25*vloss
        return {'total':total_loss,'assignment':assignment,'birth':birth,'pair':pair,'existence':existence,'visibility':vloss,
                'positive_observations':pos.sum().detach(),'NULL_observations':neg.sum().detach(),'birth_targets':(newborn&valid).sum().detach(),
                'ambiguous_masked':((identity==-2)&observed_valid).sum().detach(),
                'legal_positive_observations':((identity>=0)&valid).sum().detach(),
                'FP_observations':((identity==-1)&valid).sum().detach()}
