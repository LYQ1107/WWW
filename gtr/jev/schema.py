"""Tensor schema shared by the audit's cached choice models."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ChoiceBatch:
    query: torch.Tensor
    options: torch.Tensor
    option_mask: torch.Tensor
    target: torch.Tensor


def validate_choice_batch(batch: ChoiceBatch) -> None:
    if batch.query.ndim != 2 or batch.options.ndim != 3:
        raise ValueError("query/options must be [B,D] and [B,K,D]")
    if batch.option_mask.shape != batch.options.shape[:2] or batch.option_mask.dtype != torch.bool:
        raise ValueError("option_mask must be boolean [B,K]")
    if not bool(batch.option_mask.any(dim=1).all()):
        raise ValueError("every choice needs one valid option")
    if batch.target.shape != (batch.query.shape[0],):
        raise ValueError("target must be [B]")
